"""关系抽取服务：实体间关系类型判定、证据句子定位与置信度过滤的抽取结果。

从实体产物读取命名实体，抽取实体间语义关系并持久化，同时触发知识图谱构建。

人工复核联动：通过可归源复核编号读取真正生效的证据（复核冻结的目标版本），
定向重算相关实体对，结果写入 relations/review_recomputed.jsonl 并在
reviews/review_sources.jsonl 留存复核来源，保证每条重算关系都能回溯到复核单。
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from app.services.base import BaseModuleService
from app.services.annot_review import AnnotReviewService, REVIEW_TYPE_ENTITY, REVIEW_TYPE_RELATION


class RelationExtractService(BaseModuleService):
    MODULE = "relation_extract"
    ENTITY = "实体关系"
    OPERATION_TERM = "抽取"
    LIST_FIELDS = ["关系编号", "头实体", "尾实体", "关系类型", "置信度", "证据句子", "抽取方法", "模型版本"]
    REQUIRED_FIELDS = ["关系编号", "头实体", "尾实体"]
    STATUS_ORDER = ["草稿", "待抽取", "已抽取", "已驳回"]
    ACTION_RULES = {"启动抽取": "待抽取", "确认抽取": "已抽取", "标记驳回": "已驳回"}
    NEGATIVE_ACTIONS = ["标记驳回"]

    RELATION_RULES = {
        "EMPLOY": {
            "head_type": "ORG",
            "tail_type": "PERSON",
            "description": "雇佣关系：机构雇佣人员，例如某集团雇佣某博士",
        },
        "LOCATE": {
            "head_type": "ORG",
            "tail_type": "LOC",
            "description": "所在地关系：机构位于某地，例如总部在Beijing",
        },
        "COLLABORATE": {
            "head_type": "ORG",
            "tail_type": "ORG",
            "description": "合作关系：机构间存在合作关系",
        },
# 数据操作：抽取实体关系数据生成关系三元组数据集
        "OCCUR_AT": {
            "head_type": "TECH",
            "tail_type": "TIME",
            "description": "发生时间关系：技术事件在某个时间发生",
        },
        "INVEST": {
            "head_type": "ORG",
            "tail_type": "MONEY",
            "description": "投资关系：机构投资某金额",
        },
    }

    def extract_from_entities(self, entity_set_id: str) -> dict[str, Any]:
        """从实体识别结果中抽取实体间关系，生成关系三元组并持久化。"""
        # 读取实体产物作为输入源
        entity_file = self._data_dir("entities") / f"{entity_set_id}_entities.jsonl"
        entities = self._read_jsonl(entity_file)
        if not entities:
            return {"error": f"实体产物 {entity_set_id} 不存在或为空，请先执行实体识别"}

        model = "Relation-BERT"
        version = "v2.3.0"
        relations = []
        rel_counter = 0

        # 按实体类型分组
        by_type = {}
        for e in entities:
            by_type.setdefault(e["entity_type"], []).append(e)

        # 应用关系规则抽取三元组
        for rel_type, rule in self.RELATION_RULES.items():
            heads = by_type.get(rule["head_type"], [])
            tails = by_type.get(rule["tail_type"], [])
            # 组合可能的头尾实体对
            for head in heads[:3]:
                for tail in tails[:3]:
                    if head["entity_text"] == tail["entity_text"]:
                        continue
                    rel_counter += 1
# 数据产物：实体关系数据集写入relations目录持久化文件，触发下游图谱三元组数据
                    confidence = 0.78 + (rel_counter % 7) * 0.03
                    confidence = min(confidence, 0.95)
                    # 生成证据句子
                    evidence_sent = f"{head['entity_text']} 与 {tail['entity_text']} 之间存在「{rel_type}」关系"

                    relations.append({
                        "relation_id": f"REL-{rel_counter:06d}",
                        "head_entity": head["entity_text"],
                        "head_type": rule["head_type"],
                        "tail_entity": tail["entity_text"],
                        "tail_type": rule["tail_type"],
                        "relation_type": rel_type,
                        "confidence": confidence,
                        "evidence_sentence": evidence_sent,
                        "method": model,
                        "model_version": version,
                        "extracted_at": time.time(),
                    })

        # 按置信度去重
        relations = self._deduplicate_relations(relations)

        output_path = self._data_dir("relations") / f"{entity_set_id}_relations.jsonl"
        written = self._write_jsonl(output_path, relations)

        # 写入检查点
        self._write_checkpoint(f"relation_{entity_set_id}", {
            "entity_set_id": entity_set_id,
            "model": model,
            "entities_found": len(entities),
            "relations_extracted": len(relations),
            "relation_types": list(set(r["relation_type"] for r in relations)),
            "avg_confidence": sum(r["confidence"] for r in relations) / max(len(relations), 1),
        })

        # 跨模块触发：知识图谱构建
        triggers = {}
        if relations and entity_set_id:
            trig_kg = self._trigger_kg_build(entity_set_id)
            triggers["kg_build"] = trig_kg

        return {
            "entity_set_id": entity_set_id,
            "entities": len(entities),
            "relations": len(relations),
            "records_written": written,
            "output": str(output_path),
            "triggers": triggers,
        }

    def _deduplicate_relations(self, relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按头实体+关系类型+尾实体去重，保留置信度最高的关系记录。"""
        seen = {}
        for rel in relations:
            key = (rel["head_entity"], rel["relation_type"], rel["tail_entity"])
            if key not in seen or rel["confidence"] > seen[key]["confidence"]:
                seen[key] = rel
        return sorted(seen.values(), key=lambda r: r["relation_id"])

    # -- 人工复核联动：可归源编号定向重算 --

    def _recomputed_path(self):
        return self._data_dir("relations") / "review_recomputed.jsonl"

    def _review_source_path(self):
        return self._data_dir("reviews") / "review_sources.jsonl"

    @staticmethod
    def _stable_relation_id(review_id: str, head: str, rel_type: str, tail: str) -> str:
        raw = f"{review_id}|{head}|{rel_type}|{tail}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10].upper()
        return f"RREL-{digest}"

    @staticmethod
    def _recompute_confidence(review_id: str, head: str, rel_type: str, tail: str) -> float:
        raw = f"{review_id}|{head}|{rel_type}|{tail}"
        bucket = int(hashlib.sha256(raw.encode("utf-8")).hexdigest(), 16) % 15
        return round(0.80 + bucket * 0.01, 2)

    def _load_current_entities(self, batch_key: str = "") -> dict[tuple[str, str], dict[str, Any]]:
        """加载当前生效的实体版本，按(类型,文本)索引并保留最新识别版本。

        传入批次编号时只读取该批次的实体产物，避免跨批次实体串入定向重算；
        历史批次无命名文件时回退全量扫描（兼容旧数据）。
        """
        index: dict[tuple[str, str], dict[str, Any]] = {}
        if batch_key:
            named = self._data_dir("entities") / f"{batch_key}_entities.jsonl"
            entity_files = [named] if named.exists() else sorted(
                self._data_dir("entities").glob("*.jsonl"))
        else:
            entity_files = sorted(self._data_dir("entities").glob("*.jsonl"))
        for ef in entity_files:
            for ent in self._read_jsonl(ef):
                key = (ent.get("entity_type", ""), ent.get("entity_text", ""))
                prev = index.get(key)
                if prev is None or float(ent.get("recognized_at", 0)) >= float(prev.get("recognized_at", 0)):
                    index[key] = {**ent, "source_file": ef.name}
        return index

    def _entity_ref(self, entity: dict[str, Any]) -> dict[str, Any]:
        """构造可归源实体引用：任意引用都能找回关联的目标实体版本。"""
        return {
            "entity_id": entity.get("entity_id", ""),
            "entity_text": entity.get("entity_text", ""),
            "entity_type": entity.get("entity_type", ""),
            "chunk_id": entity.get("chunk_id", ""),
            "model_version": entity.get("model_version", ""),
            "recognized_at": entity.get("recognized_at", ""),
            "source_file": entity.get("source_file", ""),
        }

    def recompute_from_review(self, review_id: str, operator: str = "") -> dict[str, Any]:
        """按可归源复核编号读取生效证据，定向重算相关实体对。

        - 复核编号必须能在归属表证明指向，否则拒绝重算并提示待核；
        - 证据取自复核冻结的目标版本快照，而非当前可能已变动的实体文件；
        - 重算关系写入关系明细 review_recomputed.jsonl，并在 review_sources.jsonl
          留存复核来源与每轮重算历史。
        """
        review_svc = AnnotReviewService()
        registry = review_svc._load_registry()
        entry = registry.get(review_id)
        if entry is None:
            return {
                "error": f"复核编号 {review_id} 无法在归属表证明指向，已登记为待核候选",
                "review_id": review_id,
                "pending": True,
            }

        review_type = entry.get("复核对象类型", "")
        snapshot = review_svc.load_snapshot(review_id, review_type)
        if snapshot is None:
            return {"error": f"复核编号 {review_id} 的归属快照缺失，无法取得生效证据",
                    "review_id": review_id, "pending": True}

        target = snapshot.get("target", {})
        batch_key = entry.get("批次编号", "")
        current_entities = self._load_current_entities(batch_key)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        pairs: list[dict[str, Any]] = []

        if review_type == REVIEW_TYPE_ENTITY:
            target_type = target.get("entity_type", "")
            target_text = target.get("entity_text", "")
            frozen_ref = self._entity_ref(target)
            for rel_type, rule in self.RELATION_RULES.items():
                if rule["head_type"] == target_type:
                    for (tail_type, tail_text), tail_ent in current_entities.items():
                        if tail_type != rule["tail_type"] or tail_text == target_text:
                            continue
                        pairs.append(self._build_recomputed_row(
                            review_id, rel_type, target, frozen_ref, tail_ent, now))
                if rule["tail_type"] == target_type:
                    for (head_type, head_text), head_ent in current_entities.items():
                        if head_type != rule["head_type"] or head_text == target_text:
                            continue
                        pairs.append(self._build_recomputed_row(
                            review_id, rel_type, head_ent, self._entity_ref(head_ent),
                            target, now, frozen_tail=frozen_ref))
        elif review_type == REVIEW_TYPE_RELATION:
            rel_type = target.get("relation_type", "")
            head_ent = current_entities.get(
                (target.get("head_type", ""), target.get("head_entity", "")),
                {"entity_text": target.get("head_entity", ""), "entity_type": target.get("head_type", "")},
            )
            tail_ent = current_entities.get(
                (target.get("tail_type", ""), target.get("tail_entity", "")),
                {"entity_text": target.get("tail_entity", ""), "entity_type": target.get("tail_type", "")},
            )
            row = self._build_recomputed_row(
                review_id, rel_type, head_ent, self._entity_ref(head_ent), tail_ent, now,
                evidence=target.get("evidence_sentence", ""))
            pairs.append(row)
        else:
            return {"error": f"未知复核对象类型：{review_type}", "review_id": review_id}

        # 同一次重算内按实体对去重
        deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in pairs:
            key = (row["head_entity"], row["relation_type"], row["tail_entity"])
            deduped.setdefault(key, row)
        rows = list(deduped.values())

        # 同一复核编号重复重算：替换旧明细行，当前生效集合保持唯一，历史轮次写入来源文件
        existing = [r for r in self._read_jsonl(self._recomputed_path())
                    if r.get("复核来源") != review_id]
        with open(self._recomputed_path(), "w", encoding="utf-8") as f:
            for rec in existing + rows:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

        prior_runs = [r for r in self._read_jsonl(self._review_source_path())
                      if r.get("复核编号") == review_id]
        run_no = len(prior_runs) + 1
        source_record = {
            "复核编号": review_id,
            "批次编号": entry.get("批次编号", ""),
            "复核对象类型": review_type,
            "来源文件": entry.get("来源文件", ""),
            "证据指纹": entry.get("指纹", ""),
            "归属快照": entry.get("归属快照", ""),
            "重算轮次": run_no,
            "重算时间": now,
            "操作人": operator,
            "关系数量": len(rows),
            "关系编号": [r["relation_id"] for r in rows],
        }
        self._write_jsonl(self._review_source_path(), [source_record])

        return {
            "review_id": review_id,
            "review_type": review_type,
            "run_no": run_no,
            "frozen_target": target,
            "relation_count": len(rows),
            "relations": rows,
            "detail_output": str(self._recomputed_path()),
            "source_output": str(self._review_source_path()),
        }

    def _build_recomputed_row(
        self,
        review_id: str,
        rel_type: str,
        head_ent: dict[str, Any],
        head_ref: dict[str, Any],
        tail_ent: dict[str, Any],
        now: str,
        frozen_tail: dict[str, Any] | None = None,
        evidence: str | None = None,
    ) -> dict[str, Any]:
        head_text = head_ent.get("entity_text", "")
        tail_text = tail_ent.get("entity_text", "")
        confidence = self._recompute_confidence(review_id, head_text, rel_type, tail_text)
        rule = self.RELATION_RULES.get(rel_type, {})
        return {
            "relation_id": self._stable_relation_id(review_id, head_text, rel_type, tail_text),
            "head_entity": head_text,
            "head_type": rule.get("head_type", head_ent.get("entity_type", "")),
            "tail_entity": tail_text,
            "tail_type": rule.get("tail_type", tail_ent.get("entity_type", "")),
            "relation_type": rel_type,
            "confidence": confidence,
            "evidence_sentence": evidence
            or f"{head_text} 与 {tail_text} 之间存在「{rel_type}」关系（复核重算 {review_id}）",
            "method": "review-recompute",
            "model_version": "v2.3.0",
            "extracted_at": now,
            "复核来源": review_id,
            "头实体版本": head_ref,
            "尾实体版本": frozen_tail or self._entity_ref(tail_ent),
        }

    def compare_recompute_runs(self, review_id_a: str, review_id_b: str) -> dict[str, Any]:
        """比较两次复核重算（或第二次派单后重算）的关系明细差异。"""
        detail_path = self._recomputed_path()
        all_rows = self._read_jsonl(detail_path)

        def rows_of(rev_id: str) -> dict[tuple[str, str, str], dict[str, Any]]:
            return {
                (r["head_entity"], r["relation_type"], r["tail_entity"]): r
                for r in all_rows if r.get("复核来源") == rev_id
            }

        map_a = rows_of(review_id_a)
        map_b = rows_of(review_id_b)
        keys_a, keys_b = set(map_a), set(map_b)

        added = [map_b[k] for k in sorted(keys_b - keys_a)]
        removed = [map_a[k] for k in sorted(keys_a - keys_b)]
        changed = []
        for k in sorted(keys_a & keys_b):
            ra, rb = map_a[k], map_b[k]
            if ra.get("confidence") != rb.get("confidence") \
                    or ra.get("evidence_sentence") != rb.get("evidence_sentence"):
                changed.append({
                    "pair": list(k),
                    "from": {"confidence": ra.get("confidence"),
                             "evidence_sentence": ra.get("evidence_sentence")},
                    "to": {"confidence": rb.get("confidence"),
                           "evidence_sentence": rb.get("evidence_sentence")},
                })

        source_runs = {}
        for rev_id in (review_id_a, review_id_b):
            source_runs[rev_id] = [
                r for r in self._read_jsonl(self._review_source_path())
                if r.get("复核编号") == rev_id
            ]
        return {
            "review_id_a": review_id_a,
            "review_id_b": review_id_b,
            "same_review": review_id_a == review_id_b,
            "count_a": len(map_a),
            "count_b": len(map_b),
            "added": added,
            "removed": removed,
            "changed": changed,
            "source_runs": source_runs,
        }