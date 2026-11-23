"""人工复核服务：低置信抽取结果的人工复核任务、复核结论登记与退回原因追溯。

从抽取质量报告筛选需要人工复核的低置信度条目，创建复核任务并记录结论。

归属校验设计（修复再次派单串号/证据错位/结论串位）：
- 复核编号不再使用易受文件排序影响的顺序号，而是由「复核对象类型 + 证据指纹」
  哈希生成的可归源编号（REV-ENT-XXXX / REV-REL-XXXX），同一证据无论文件如何
  重排、无论派单多少次都得到同一编号，不同来源的证据必然得到不同编号。
- 每条复核编号在 reviews/review_registry.jsonl 登记一条归属记录，保存来源文件、
  证据指纹，并把目标实体/关系记录冻结为版本快照（reviews/snapshots/）。
- 历史数据不覆盖：再次派单只累加派单次数，旧结论进入「结论历史」，冻结快照
  永久保留旧结论当时所依据的实体版本。
- 完全相同的证据（指纹相同）全局只登记一次；无法在归属表证明指向的历史编号
  不允许冒领，统一落入 reviews/pending_candidates.jsonl 作为待核候选。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from app.services.base import BaseModuleService

REVIEW_TYPE_ENTITY = "实体"
REVIEW_TYPE_RELATION = "关系"
_TYPE_TOKEN = {REVIEW_TYPE_ENTITY: "ENT", REVIEW_TYPE_RELATION: "REL"}


class AnnotReviewService(BaseModuleService):
    MODULE = "annot_review"
    ENTITY = "复核任务"
    OPERATION_TERM = "校验"
    LIST_FIELDS = ["复核编号", "分块编号", "复核对象", "原判定", "复核结论", "复核人", "复核时间", "退回原因"]
    REQUIRED_FIELDS = ["复核编号", "分块编号", "复核对象"]
    STATUS_ORDER = ["草稿", "待复核", "复核中", "已结案"]
    ACTION_RULES = {"分派复核": "待复核", "提交结论": "已结案", "标记退回": "已结案"}
    NEGATIVE_ACTIONS = []

    # -- 归属存储路径 --

    def _registry_path(self) -> Path:
        return self._data_dir("reviews") / "review_registry.jsonl"

    def _pending_path(self) -> Path:
        return self._data_dir("reviews") / "pending_candidates.jsonl"

    def _snapshot_dir(self) -> Path:
        return self._data_dir("reviews") / "snapshots"

    def _snapshot_path(self, review_id: str, review_type: str) -> Path:
        token = _TYPE_TOKEN.get(review_type, "GEN")
        return self._snapshot_dir() / f"{review_id}_{token.lower()}.json"

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        """读取复核编号归属表，以复核编号为键。"""
        registry: dict[str, dict[str, Any]] = {}
        for rec in self._read_jsonl(self._registry_path()):
            rev_id = rec.get("复核编号")
            if rev_id:
                registry[rev_id] = rec
        return registry

    def _save_registry(self, registry: dict[str, dict[str, Any]]) -> None:
        """整体回写归属表（仅归属表使用全量写，业务任务清单仍为追加写）。"""
        path = self._registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in registry.values():
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    # -- 指纹与可归源编号 --

    @staticmethod
    def _entity_fingerprint(entity: dict[str, Any]) -> str:
        return "|".join([
            "ENT",
            str(entity.get("entity_type", "")),
            str(entity.get("entity_text", "")),
            str(entity.get("chunk_id", "")),
            f"{entity.get('start_offset', '')}-{entity.get('end_offset', '')}",
        ])

    @staticmethod
    def _relation_fingerprint(relation: dict[str, Any]) -> str:
        return "|".join([
            "REL",
            str(relation.get("head_entity", "")),
            str(relation.get("relation_type", "")),
            str(relation.get("tail_entity", "")),
            str(relation.get("evidence_sentence", "")),
        ])

    def _make_review_id(self, review_type: str, fingerprint: str) -> str:
        token = _TYPE_TOKEN.get(review_type, "GEN")
        digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:10].upper()
        return f"REV-{token}-{digest}"

    def _freeze_snapshot(self, review_id: str, review_type: str, record: dict[str, Any]) -> str:
        """冻结目标实体/关系记录为版本快照，返回快照文件名（相对reviews目录）。"""
        snap_dir = self._snapshot_dir()
        snap_dir.mkdir(parents=True, exist_ok=True)
        path = self._snapshot_path(review_id, review_type)
        payload = {
            "复核编号": review_id,
            "复核对象类型": review_type,
            "冻结时间": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target": record,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        return str(path.relative_to(self._data_dir("reviews")))

    def load_snapshot(self, review_id: str, review_type: str) -> dict[str, Any] | None:
        """读取复核编号冻结的目标版本；快照缺失返回None（归属无法证明）。"""
        path = self._snapshot_path(review_id, review_type)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -- 批次输入（只读本批次来源文件，修复跨批次串号） --

    def _load_batch_records(self, sub_dir: str, batch_key: str, suffix: str) -> tuple[list[dict[str, Any]], str]:
        """优先读取本批次产物文件；历史批次无命名文件时回退全量扫描。"""
        named = self._data_dir(sub_dir) / f"{batch_key}{suffix}"
        if named.exists():
            return self._read_jsonl(named), named.name
        legacy = [rec for fp in sorted(self._data_dir(sub_dir).glob("*.jsonl"))
                  for rec in self._read_jsonl(fp)]
        source = named.name if not legacy else "legacy-scan"
        return legacy, source

    # -- 派单 --

    def create_review_tasks(self, batch_key: str, threshold: float = 0.80) -> dict[str, Any]:
        """从低置信度抽取结果创建人工复核任务。

        编号由证据指纹派生且带归属登记；同一证据重复派单只记一次，
        已登记编号再次派单累加派单次数而不产生新编号。
        """
        entities, entity_source = self._load_batch_records("entities", batch_key, "_entities.jsonl")
        relations, relation_source = self._load_batch_records("relations", batch_key, "_relations.jsonl")

        registry = self._load_registry()
        existing_fingerprints = {rec.get("指纹") for rec in registry.values()}
        seen_this_run: set[str] = set()

        tasks: list[dict[str, Any]] = []
        new_count = 0
        duplicate_count = 0
        redispatched: list[str] = []
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        def build_and_register(
            review_type: str,
            record: dict[str, Any],
            confidence: float,
            chunk_id: str,
            display: str,
            source_file: str,
        ) -> None:
            nonlocal new_count, duplicate_count
            if review_type == REVIEW_TYPE_ENTITY:
                fingerprint = self._entity_fingerprint(record)
            else:
                fingerprint = self._relation_fingerprint(record)

            # 重复提交完全相同的证据：全局只登记一次
            if fingerprint in seen_this_run or fingerprint in existing_fingerprints:
                duplicate_count += 1
                rev_id = self._make_review_id(review_type, fingerprint)
                entry = registry.get(rev_id)
                if entry is not None:
                    entry["派单次数"] = int(entry.get("派单次数", 1)) + 1
                    entry["最近派单时间"] = now
                    entry["来源文件"] = source_file
                    redispatched.append(rev_id)
                return
            seen_this_run.add(fingerprint)

            rev_id = self._make_review_id(review_type, fingerprint)
            snapshot_ref = self._freeze_snapshot(rev_id, review_type, record)

            registry[rev_id] = {
                "复核编号": rev_id,
                "批次编号": batch_key,
                "复核对象类型": review_type,
                "来源文件": source_file,
                "指纹": fingerprint,
                "证据内容": record.get("evidence_sentence")
                if review_type == REVIEW_TYPE_RELATION else record.get("entity_text", ""),
                "归属快照": snapshot_ref,
                "首次派单时间": now,
                "最近派单时间": now,
                "派单次数": 1,
                "状态": "待复核",
                "复核结论": "",
                "复核人": "",
                "复核时间": "",
                "结论历史": [],
            }

            tasks.append({
                "复核编号": rev_id,
                "批次编号": batch_key,
                "分块编号": chunk_id,
                "复核对象": display,
                "复核对象类型": review_type,
                "原判定": f"置信度 {confidence:.2f}",
                "证据指纹": fingerprint,
                "来源文件": source_file,
                "归属快照": snapshot_ref,
                "复核结论": "",
                "复核人": "",
                "复核时间": "",
                "退回原因": "",
                "status": "待复核",
            })
            new_count += 1

        for e in entities:
            confidence = float(e.get("confidence", 1.0))
            if confidence < threshold:
                build_and_register(
                    REVIEW_TYPE_ENTITY,
                    e,
                    confidence,
                    str(e.get("chunk_id", "")),
                    f"实体: {e.get('entity_text')} [{e.get('entity_type')}]",
                    entity_source,
                )

        for r in relations:
            confidence = float(r.get("confidence", 1.0))
            if confidence < threshold:
                build_and_register(
                    REVIEW_TYPE_RELATION,
                    r,
                    confidence,
                    "",
                    f"关系: {r.get('head_entity')}→{r.get('relation_type')}→{r.get('tail_entity')}",
                    relation_source,
                )

        # 归属表整体回写；任务清单仅追加新任务，历史任务行保持原位不串位
        self._save_registry(registry)
        output_path = self._data_dir("reviews") / f"{batch_key}_reviews.jsonl"
        written = self._write_jsonl(output_path, tasks) if tasks else 0

        self._write_checkpoint(f"review_{batch_key}", {
            "batch_key": batch_key,
            "threshold": threshold,
            "tasks_created": new_count,
            "duplicate_evidence": duplicate_count,
            "redispatched": redispatched,
            "entities_reviewed": sum(1 for t in tasks if t["复核对象类型"] == REVIEW_TYPE_ENTITY),
            "relations_reviewed": sum(1 for t in tasks if t["复核对象类型"] == REVIEW_TYPE_RELATION),
        })

        return {
            "batch_key": batch_key,
            "threshold": threshold,
            "tasks_created": new_count,
            "duplicate_evidence": duplicate_count,
            "redispatched": redispatched,
            "records_written": written,
            "output": str(output_path),
            "registry": str(self._registry_path()),
        }

    # -- 结论提交（带归属校验，旧结论入历史） --

    def submit_review_conclusion(self, review_id: str, conclusion: str, reviewer: str) -> dict[str, Any]:
        """提交复核结论。

        归属表中存在的编号先做指纹/快照归属校验，通过后只更新对应编号的行，
        旧结论保留在「结论历史」；归属表无法证明指向的编号落入待核候选，
        不允许覆盖任何已有复核单。
        """
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        registry = self._load_registry()
        entry = registry.get(review_id)

        if entry is None:
            return self._append_pending_candidate(review_id, conclusion, reviewer, now,
                                                   "历史编号无法在归属表证明指向")

        # 归属校验：任务清单中的指纹必须与归属表一致，冻结快照必须存在
        review_type = entry.get("复核对象类型", "")
        snapshot = self.load_snapshot(review_id, review_type)
        if snapshot is None:
            return self._append_pending_candidate(review_id, conclusion, reviewer, now,
                                                   f"归属快照缺失：{entry.get('归属快照', '')}")

        mismatched_files = self._find_fingerprint_mismatches(review_id, entry.get("指纹", ""))
        if mismatched_files:
            return self._append_pending_candidate(review_id, conclusion, reviewer, now,
                                                   f"证据指纹与归属表不一致：{mismatched_files}")

        # 旧结论先入历史，再写入新结论，历史数据不丢失
        if entry.get("复核结论"):
            entry.setdefault("结论历史", []).append({
                "复核结论": entry["复核结论"],
                "复核人": entry.get("复核人", ""),
                "复核时间": entry.get("复核时间", ""),
            })
        entry["复核结论"] = conclusion
        entry["复核人"] = reviewer
        entry["复核时间"] = now
        entry["状态"] = "已结案"
        self._save_registry(registry)

        # 同步更新各任务清单中同一编号的行（按编号定位，不依赖行序）
        updated_files = self._update_task_rows(review_id, conclusion, reviewer, now)

        return {
            "review_id": review_id,
            "conclusion": conclusion,
            "reviewer": reviewer,
            "status": "已结案",
            "snapshot": entry.get("归属快照", ""),
            "conclusion_history": entry.get("结论历史", []),
            "updated_files": updated_files,
        }

    def _find_fingerprint_mismatches(self, review_id: str, expected_fp: str) -> list[str]:
        bad: list[str] = []
        for rf in sorted(self._data_dir("reviews").glob("*_reviews.jsonl")):
            for task in self._read_jsonl(rf):
                if task.get("复核编号") == review_id and task.get("证据指纹") != expected_fp:
                    bad.append(rf.name)
                    break
        return bad

    def _update_task_rows(
        self, review_id: str, conclusion: str, reviewer: str, now: str,
    ) -> list[str]:
        updated: list[str] = []
        for rf in sorted(self._data_dir("reviews").glob("*_reviews.jsonl")):
            tasks = self._read_jsonl(rf)
            touched = False
            for task in tasks:
                if task.get("复核编号") == review_id:
                    task["复核结论"] = conclusion
                    task["复核人"] = reviewer
                    task["复核时间"] = now
                    task["status"] = "已结案"
                    touched = True
            if touched:
                with open(rf, "w", encoding="utf-8") as f:
                    for t in tasks:
                        f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
                updated.append(rf.name)
        return updated

    def _append_pending_candidate(
        self, review_id: str, conclusion: str, reviewer: str, now: str, reason: str,
    ) -> dict[str, Any]:
        candidate = {
            "复核编号": review_id,
            "复核结论": conclusion,
            "复核人": reviewer,
            "提交时间": now,
            "待核原因": reason,
            "状态": "待核",
        }
        self._write_jsonl(self._pending_path(), [candidate])
        return {
            "review_id": review_id,
            "pending_candidate": True,
            "status": "待核",
            "reason": reason,
            "candidate_file": str(self._pending_path()),
        }

    # -- 详情与待核候选 --

    def get_review_detail(self, review_id: str) -> dict[str, Any]:
        """打开复核单：返回归属记录、冻结证据快照、任务行与重算来源，证据可对回来源。"""
        registry = self._load_registry()
        entry = registry.get(review_id)
        if entry is None:
            candidates = [c for c in self._read_jsonl(self._pending_path())
                          if c.get("复核编号") == review_id]
            return {"review_id": review_id, "found": False, "pending_candidates": candidates}

        review_type = entry.get("复核对象类型", "")
        snapshot = self.load_snapshot(review_id, review_type)
        task_rows = []
        for rf in sorted(self._data_dir("reviews").glob("*_reviews.jsonl")):
            for task in self._read_jsonl(rf):
                if task.get("复核编号") == review_id:
                    task_rows.append({"file": rf.name, **task})

        recompute_sources = [
            rec for rec in self._read_jsonl(self._data_dir("reviews") / "review_sources.jsonl")
            if rec.get("复核编号") == review_id
        ]
        return {
            "review_id": review_id,
            "found": True,
            "registry": entry,
            "snapshot": snapshot,
            "task_rows": task_rows,
            "recompute_sources": recompute_sources,
        }

    def list_pending_candidates(self) -> list[dict[str, Any]]:
        """列出所有无法证明指向、等待人工核实的候选编号。"""
        return self._read_jsonl(self._pending_path())

    def get_review_trace(self, review_id: str) -> dict[str, Any]:
        """复核编号全链路视图：派单次数、结论历史、冻结实体版本与关系重算结果。"""
        detail = self.get_review_detail(review_id)
        if not detail.get("found"):
            return detail
        recomputed = [
            rec for rec in self._read_jsonl(self._data_dir("relations") / "review_recomputed.jsonl")
            if rec.get("复核来源") == review_id
        ]
        return {**detail, "recomputed_relations": recomputed}
