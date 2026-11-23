"""人工复核可归源编号与关系重算的归属校验集成测试。

复现并覆盖问题场景：
1. 同一批先有实体乙并已派单，新文件排序靠前后新增实体甲，再次派单不得串号；
2. 旧复核单仍在用户手上，输入同一编号重算时两个来源不得混淆；
3. 已确认引用可找回冻结的目标实体版本，历史结论所用实体不丢失；
4. 完全相同证据重复提交只记一次；
5. 无法证明指向的历史编号落入待核候选；
6. 关系抽取按可归源编号读取生效证据、定向重算并写入关系明细与复核来源；
7. 比较第二次派单与从原编号发起的关系重算结果。
"""
import json
import os
import sys
import uuid

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from app.services.annot_review import AnnotReviewService
from app.services.base import DATA_ROOT
from app.services.relation_extract import RelationExtractService


def _uid(prefix="B"):
    return f"{prefix}-{uuid.uuid4().hex[:10].upper()}"


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _entity(entity_id, ent_type, text, confidence, chunk_id, source_file):
    return {
        "entity_id": entity_id,
        "entity_type": ent_type,
        "entity_text": text,
        "chunk_id": chunk_id,
        "start_offset": 0,
        "end_offset": len(text),
        "confidence": confidence,
        "model": "NER-BiLSTM-CRF",
        "model_version": "v3.1.0",
        "recognized_at": 1_700_000_000.0,
        "source_file": source_file,
    }


def _relation(rel_id, head, head_type, rel_type, tail, tail_type, confidence, evidence):
    return {
        "relation_id": rel_id,
        "head_entity": head,
        "head_type": head_type,
        "tail_entity": tail,
        "tail_type": tail_type,
        "relation_type": rel_type,
        "confidence": confidence,
        "evidence_sentence": evidence,
        "method": "Relation-BERT",
        "model_version": "v2.3.0",
        "extracted_at": 1_700_000_000.0,
    }


def _write_batch_inputs(batch_key, entities, relations):
    _write_jsonl(DATA_ROOT / "entities" / f"{batch_key}_entities.jsonl", entities)
    _write_jsonl(DATA_ROOT / "relations" / f"{batch_key}_relations.jsonl", relations)


def _review_ids(batch_key):
    path = DATA_ROOT / "reviews" / f"{batch_key}_reviews.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["复核编号"] for line in f if line.strip()]


class TestReviewProvenance:
    """可归源复核编号：派单归属、重排不串号、证据快照与结论历史。"""

    def test_redispatch_same_id_when_file_reordered_with_new_entity(self):
        """新文件排到前面并新增实体甲：实体乙两次派单编号保持一致，实体甲得到独立编号。"""
        batch_yi = _uid("BATCH-YI")
        batch_second = _uid("BATCH-2ND")
        svc = AnnotReviewService()

        yi = _entity("ENT-YI", "ORG", f"乙集团{batch_yi}", 0.55, f"CHUNK-YI-{batch_yi}", "yi.jsonl")
        jia = _entity("ENT-JIA", "ORG", f"甲集团{batch_second}", 0.50, f"CHUNK-JIA-{batch_second}", "jia.jsonl")

        # 首次派单：只有实体乙
        _write_batch_inputs(batch_yi, [yi], [])
        first = svc.create_review_tasks(batch_yi, threshold=0.8)
        assert first["tasks_created"] == 1
        first_ids = _review_ids(batch_yi)
        assert len(first_ids) == 1 and first_ids[0].startswith("REV-ENT-")

        # 第二次派单：新文件（含实体甲）排序靠前，实体乙仍在
        _write_batch_inputs(batch_second, [jia, yi], [])
        second = svc.create_review_tasks(batch_second, threshold=0.8)

        # 实体乙是相同证据，只登记一次；实体甲为新证据新增一单
        assert second["tasks_created"] == 1
        assert second["duplicate_evidence"] == 1
        assert second["redispatched"] == first_ids

        second_ids = _review_ids(batch_second)
        assert second_ids == [svc._make_review_id("实体", svc._entity_fingerprint(jia))]
        assert first_ids[0] not in second_ids

        # 归属表中实体乙编号仍然唯一且派单次数累加
        registry = svc._load_registry()
        yi_entry = registry[first_ids[0]]
        assert yi_entry["派单次数"] == 2
        assert yi_entry["批次编号"] == batch_yi
        assert registry[second_ids[0]]["派单次数"] == 1

    def test_review_id_stable_under_reordering(self):
        """打乱实体顺序后派生编号完全一致，不再依赖行序计数。"""
        svc = AnnotReviewService()
        e1 = _entity("E1", "ORG", f"甲{uuid.uuid4().hex[:6]}", 0.4, "C1", "f.jsonl")
        e2 = _entity("E2", "PERSON", f"乙{uuid.uuid4().hex[:6]}", 0.4, "C2", "f.jsonl")
        ids_a = [svc._make_review_id("实体", svc._entity_fingerprint(e)) for e in [e1, e2]]
        ids_b = [svc._make_review_id("实体", svc._entity_fingerprint(e)) for e in [e2, e1]]
        assert set(ids_a) == set(ids_b)
        assert ids_a[0] != ids_a[1]

    def test_different_sources_same_legacy_number_do_not_collide(self):
        """两个不同来源即使携带相同历史编号，派生编号也不同，输入同一编号不会混淆。"""
        svc = AnnotReviewService()
        e1 = _entity("E1", "ORG", f"来源甲{uuid.uuid4().hex[:6]}", 0.5, "C1", "a.jsonl")
        e2 = _entity("E1", "ORG", f"来源乙{uuid.uuid4().hex[:6]}", 0.5, "C2", "b.jsonl")
        id1 = svc._make_review_id("实体", svc._entity_fingerprint(e1))
        id2 = svc._make_review_id("实体", svc._entity_fingerprint(e2))
        assert id1 != id2

    def test_conclusion_keeps_history_and_frozen_entity_version(self):
        """提交结论后可通过编号找回冻结实体版本；改判后旧结论进入历史，实体版本不丢失。"""
        batch = _uid("BATCH-CONC")
        svc = AnnotReviewService()
        ent = _entity("ENT-C", "ORG", f"丙集团{batch}", 0.6, f"CHUNK-C-{batch}", "c.jsonl")
        _write_batch_inputs(batch, [ent], [])
        created = svc.create_review_tasks(batch, threshold=0.8)
        review_id = _review_ids(batch)[0]
        assert created["tasks_created"] == 1

        first = svc.submit_review_conclusion(review_id, "确认实体正确", "张三")
        assert first["status"] == "已结案"
        assert first["conclusion_history"] == []

        # 用户拿着旧复核单改判
        second = svc.submit_review_conclusion(review_id, "修正为误抽，应驳回", "李四")
        assert second["status"] == "已结案"
        assert len(second["conclusion_history"]) == 1
        assert second["conclusion_history"][0]["复核结论"] == "确认实体正确"
        assert second["conclusion_history"][0]["复核人"] == "张三"

        detail = svc.get_review_detail(review_id)
        assert detail["found"] is True
        # 冻结快照保留旧结论所用的实体
        assert detail["snapshot"]["target"]["entity_text"] == ent["entity_text"]
        assert detail["snapshot"]["target"]["entity_id"] == "ENT-C"
        assert detail["registry"]["复核结论"] == "修正为误抽，应驳回"
        # 任务清单行与详情证据一致，不再错位
        assert all(row["证据指纹"] == detail["registry"]["指纹"] for row in detail["task_rows"])

    def test_unknown_legacy_number_becomes_pending_candidate(self):
        """归属表无法证明指向的历史编号不能冒领，登记为待核候选。"""
        svc = AnnotReviewService()
        legacy_id = f"REV-000{uuid.uuid4().hex[:6].upper()}"
        result = svc.submit_review_conclusion(legacy_id, "用户拿着旧单提交", "王五")
        assert result["pending_candidate"] is True
        assert result["status"] == "待核"

        candidates = svc.list_pending_candidates()
        mine = [c for c in candidates if c["复核编号"] == legacy_id]
        assert len(mine) == 1 and mine[0]["待核原因"]

        detail = svc.get_review_detail(legacy_id)
        assert detail["found"] is False
        assert len(detail["pending_candidates"]) == 1

    def test_identical_evidence_recorded_once_globally(self):
        """完全相同的证据在不同批次重复派单也只登记一次。"""
        batch_a = _uid("BATCH-DUP-A")
        batch_b = _uid("BATCH-DUP-B")
        svc = AnnotReviewService()
        ent = _entity("ENT-DUP", "LOC", f"Beijing园区{batch_a}", 0.3, "CHUNK-DUP", "dup.jsonl")

        _write_batch_inputs(batch_a, [ent], [])
        first = svc.create_review_tasks(batch_a, threshold=0.8)
        assert first["tasks_created"] == 1

        _write_batch_inputs(batch_b, [ent], [])
        second = svc.create_review_tasks(batch_b, threshold=0.8)
        assert second["tasks_created"] == 0
        assert second["duplicate_evidence"] == 1
        # 第二个批次的任务清单不产生重复行
        assert not (DATA_ROOT / "reviews" / f"{batch_b}_reviews.jsonl").exists()

        review_id = _review_ids(batch_a)[0]
        registry = svc._load_registry()
        same_ids = [rid for rid, rec in registry.items()
                    if rec.get("指纹") == svc._entity_fingerprint(ent)]
        assert same_ids == [review_id]

    def test_relation_review_tasks_also_traceable(self):
        """关系类低置信条目同样生成可归源编号并冻结关系快照。"""
        batch = _uid("BATCH-REL")
        svc = AnnotReviewService()
        rel = _relation("REL-X", f"集团{batch}", "ORG", "LOCATE", f"地点{batch}", "LOC", 0.42,
                        f"{batch} 总部位于 Beijing")
        _write_batch_inputs(batch, [], [rel])
        created = svc.create_review_tasks(batch, threshold=0.8)
        assert created["tasks_created"] == 1
        review_id = _review_ids(batch)[0]
        assert review_id.startswith("REV-REL-")
        snapshot = svc.load_snapshot(review_id, "关系")
        assert snapshot["target"]["relation_id"] == "REL-X"
        assert snapshot["target"]["evidence_sentence"] == rel["evidence_sentence"]


class TestRelationRecomputeFromReview:
    """关系抽取按可归源复核编号读取生效证据并定向重算。"""

    def _seed_entity_batch(self, batch):
        target = _entity("ENT-T", "ORG", f"目标集团{batch}", 0.5, f"CHUNK-T-{batch}", "t.jsonl")
        loc = _entity("ENT-L", "LOC", f"Beijing基地{batch}", 0.9, f"CHUNK-L-{batch}", "l.jsonl")
        person = _entity("ENT-P", "PERSON", f"赵博士{batch}", 0.9, f"CHUNK-P-{batch}", "p.jsonl")
        _write_batch_inputs(batch, [target, loc, person], [])
        return target, loc, person

    def test_recompute_from_entity_review_writes_detail_and_source(self):
        batch = _uid("BATCH-REC")
        review_svc = AnnotReviewService()
        rel_svc = RelationExtractService()
        target, loc, person = self._seed_entity_batch(batch)

        review_svc.create_review_tasks(batch, threshold=0.8)
        review_id = review_svc._make_review_id(
            "实体", review_svc._entity_fingerprint(target))

        result = rel_svc.recompute_from_review(review_id, operator="复核员甲")
        assert "error" not in result
        assert result["relation_count"] >= 1
        # 只重算与被复核实体相关的实体对
        pairs = {(r["head_entity"], r["relation_type"], r["tail_entity"]) for r in result["relations"]}
        assert all(target["entity_text"] in (p[0], p[2]) for p in pairs)
        assert (target["entity_text"], "LOCATE", loc["entity_text"]) in pairs

        # 关系明细落盘且带复核来源与可归源实体版本引用
        detail_path = DATA_ROOT / "relations" / "review_recomputed.jsonl"
        rows = [json.loads(line) for line in detail_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        mine = [r for r in rows if r["复核来源"] == review_id]
        assert len(mine) == result["relation_count"]
        for row in mine:
            assert row["method"] == "review-recompute"
            assert row["头实体版本"]["entity_text"]
            assert row["尾实体版本"]["entity_text"]
            assert row["头实体版本"]["source_file"]

        # 复核来源登记
        sources = [json.loads(line) for line in
                   (DATA_ROOT / "reviews" / "review_sources.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        src = [s for s in sources if s["复核编号"] == review_id]
        assert len(src) == 1 and src[0]["重算轮次"] == 1
        assert src[0]["证据指纹"] == review_svc._entity_fingerprint(target)

    def test_recompute_uses_frozen_evidence_not_current_file(self):
        """重算依据复核冻结快照：即使当前实体文件变化，目标版本仍可找回。"""
        batch = _uid("BATCH-FRZ")
        review_svc = AnnotReviewService()
        rel_svc = RelationExtractService()
        target, loc, _ = self._seed_entity_batch(batch)

        review_svc.create_review_tasks(batch, threshold=0.8)
        review_id = review_svc._make_review_id("实体", review_svc._entity_fingerprint(target))

        # 当前实体文件被改写（模拟新版本覆盖），冻结快照仍是旧版本
        changed = {**target, "entity_text": f"目标集团改名{batch}", "recognized_at": 1_800_000_000.0}
        _write_jsonl(DATA_ROOT / "entities" / f"{batch}_entities.jsonl", [changed, loc])

        result = rel_svc.recompute_from_review(review_id)
        assert result["frozen_target"]["entity_text"] == target["entity_text"]
        detail = review_svc.get_review_detail(review_id)
        assert detail["snapshot"]["target"]["entity_text"] == target["entity_text"]

    def test_recompute_rejects_unprovable_review_id(self):
        rel_svc = RelationExtractService()
        result = rel_svc.recompute_from_review(f"REV-FAKE-{uuid.uuid4().hex[:8].upper()}")
        assert result.get("pending") is True
        assert "error" in result

    def test_second_dispatch_vs_original_id_recompute_comparison(self):
        """完整重放：首次派单→关系重算；新文件排前新增实体甲后第二次派单；
        再从原编号发起重算，两次结果可比较且原编号证据不混淆。"""
        batch_one = _uid("BATCH-ONE")
        batch_two = _uid("BATCH-TWO")
        review_svc = AnnotReviewService()
        rel_svc = RelationExtractService()

        yi_org = _entity("E-YI", "ORG", f"乙集团{batch_one}", 0.5, f"CHUNK-YI-{batch_one}", "yi.jsonl")
        yi_loc = _entity("E-YILOC", "LOC", f"Shanghai分部{batch_one}", 0.9, f"CHUNK-LOC-{batch_one}", "loc.jsonl")
        jia_org = _entity("E-JIA", "ORG", f"甲集团{batch_two}", 0.5, f"CHUNK-JIA-{batch_two}", "jia.jsonl")

        # 首次派单：实体乙
        _write_batch_inputs(batch_one, [yi_org, yi_loc], [])
        review_svc.create_review_tasks(batch_one, threshold=0.8)
        yi_review_id = review_svc._make_review_id("实体", review_svc._entity_fingerprint(yi_org))

        run1 = rel_svc.recompute_from_review(yi_review_id, operator="首次重算")
        assert run1["run_no"] == 1
        run1_pairs = {(r["head_entity"], r["relation_type"], r["tail_entity"]) for r in run1["relations"]}

        # 第二次派单：新文件排前面并增加实体甲，实体乙同编号复用
        _write_batch_inputs(batch_two, [jia_org, yi_org, yi_loc], [])
        second = review_svc.create_review_tasks(batch_two, threshold=0.8)
        assert second["redispatched"] == [yi_review_id]
        jia_review_id = _review_ids(batch_two)[0]
        assert jia_review_id != yi_review_id

        # 从原编号再次发起关系重算：证据仍指向实体乙，轮次累加，明细按编号替换
        run2 = rel_svc.recompute_from_review(yi_review_id, operator="二次重算")
        assert run2["run_no"] == 2
        run2_pairs = {(r["head_entity"], r["relation_type"], r["tail_entity"]) for r in run2["relations"]}
        assert run1_pairs == run2_pairs
        # 原编号重算只产出实体乙所在的实体对，且不会把实体甲当作被复核目标
        assert all(yi_org["entity_text"] in (p[0], p[2]) for p in run2_pairs)

        # 实体甲独立重算，两个来源互不混淆；甲乙合作对可在甲的重算中出现，
        # 但绝不会出现在乙的原编号重算里
        run_jia = rel_svc.recompute_from_review(jia_review_id)
        assert run_jia["review_id"] == jia_review_id
        jia_pairs = {(r["head_entity"], r["relation_type"], r["tail_entity"]) for r in run_jia["relations"]}
        assert all(jia_org["entity_text"] in (p[0], p[2]) for p in jia_pairs)
        assert jia_pairs.isdisjoint(run2_pairs)

        # 比较第二次派单后的重算与原编号重算结果
        compare = rel_svc.compare_recompute_runs(jia_review_id, yi_review_id)
        assert compare["count_a"] == run_jia["relation_count"]
        assert compare["count_b"] == run2["relation_count"]
        added_pairs = {(r["head_entity"], r["relation_type"], r["tail_entity"]) for r in compare["removed"]}
        removed_pairs = {(r["head_entity"], r["relation_type"], r["tail_entity"]) for r in compare["added"]}
        # 两个编号的实体对集合互不重叠，来源不混淆
        assert all(jia_org["entity_text"] in p for p in added_pairs)
        assert all(yi_org["entity_text"] in p for p in removed_pairs)

        # 复核血缘视图可完整追溯
        trace = review_svc.get_review_trace(yi_review_id)
        assert trace["found"] is True
        assert trace["registry"]["派单次数"] == 2
        assert len(trace["recompute_sources"]) == 2
        assert len(trace["recomputed_relations"]) == run2["relation_count"]

    def test_recompute_idempotent_replaces_same_review_rows(self):
        """同一复核编号重复重算：明细中该编号的生效集合唯一，历史轮次保留在来源文件。"""
        batch = _uid("BATCH-IDEM")
        review_svc = AnnotReviewService()
        rel_svc = RelationExtractService()
        target, loc, _ = self._seed_entity_batch(batch)
        review_svc.create_review_tasks(batch, threshold=0.8)
        review_id = review_svc._make_review_id("实体", review_svc._entity_fingerprint(target))

        r1 = rel_svc.recompute_from_review(review_id)
        r2 = rel_svc.recompute_from_review(review_id)
        assert r2["run_no"] == 2

        detail_rows = [json.loads(line) for line in
                       (DATA_ROOT / "relations" / "review_recomputed.jsonl").read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        mine = [r for r in detail_rows if r["复核来源"] == review_id]
        # 同一实体对只保留一份生效明细，关系编号稳定
        assert len(mine) == r2["relation_count"] == r1["relation_count"]
        ids_run1 = {r["relation_id"] for r in r1["relations"]}
        ids_run2 = {r["relation_id"] for r in r2["relations"]}
        assert ids_run1 == ids_run2

    def test_relation_review_recompute_targets_single_pair_with_original_evidence(self):
        """关系复核单重算只定向重算该关系对，并沿用冻结快照中的原始证据句。"""
        batch = _uid("BATCH-RREC")
        review_svc = AnnotReviewService()
        rel_svc = RelationExtractService()
        head = _entity("RH", "ORG", f"关系头集团{batch}", 0.9, "CH1", "h.jsonl")
        tail = _entity("RT", "LOC", f"关系尾地点{batch}", 0.9, "CH2", "t.jsonl")
        rel = _relation("RR1", head["entity_text"], "ORG", "LOCATE", tail["entity_text"],
                        "LOC", 0.4, f"{batch} 原始证据句")
        _write_batch_inputs(batch, [head, tail], [rel])
        review_svc.create_review_tasks(batch, threshold=0.8)
        review_id = review_svc._make_review_id("关系", review_svc._relation_fingerprint(rel))

        result = rel_svc.recompute_from_review(review_id, operator="关系复核员")
        assert result["review_type"] == "关系"
        assert result["relation_count"] == 1
        row = result["relations"][0]
        assert (row["head_entity"], row["relation_type"], row["tail_entity"]) == \
               (head["entity_text"], "LOCATE", tail["entity_text"])
        assert row["evidence_sentence"] == f"{batch} 原始证据句"
        assert row["复核来源"] == review_id
