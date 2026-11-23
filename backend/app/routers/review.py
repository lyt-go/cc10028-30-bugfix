"""人工复核与关系重算路由：围绕可归源复核编号的派单、结论、待核与重算接口。"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.annot_review import AnnotReviewService
from app.services.relation_extract import RelationExtractService

router = APIRouter(prefix="/api/annot_review", tags=["人工复核"])


class DispatchPayload(BaseModel):
    batch_key: str
    threshold: float = Field(default=0.80, ge=0.0, le=1.0)


class ConclusionPayload(BaseModel):
    review_id: str
    conclusion: str
    reviewer: str


class RecomputePayload(BaseModel):
    review_id: str
    operator: str = ""


class ComparePayload(BaseModel):
    review_id_a: str
    review_id_b: str


@router.post("/dispatch")
def dispatch_tasks(payload: DispatchPayload) -> dict[str, Any]:
    """按批次创建复核任务（再次派单同编号复用、同证据只记一次）。"""
    return AnnotReviewService().create_review_tasks(payload.batch_key, payload.threshold)


@router.post("/conclusion")
def submit_conclusion(payload: ConclusionPayload) -> dict[str, Any]:
    """提交复核结论；归属无法证明时返回待核候选而非覆盖旧单。"""
    result = AnnotReviewService().submit_review_conclusion(
        payload.review_id, payload.conclusion, payload.reviewer)
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/{review_id}/detail")
def review_detail(review_id: str) -> dict[str, Any]:
    """复核详情：归属记录、冻结证据快照、任务行与重算来源。"""
    return AnnotReviewService().get_review_detail(review_id)


@router.get("/pending/candidates")
def pending_candidates() -> dict[str, Any]:
    """待核候选编号清单。"""
    items = AnnotReviewService().list_pending_candidates()
    return {"items": items, "total": len(items)}


@router.post("/recompute")
def recompute(payload: RecomputePayload) -> dict[str, Any]:
    """按复核编号定向重算相关实体对，写入关系明细与复核来源。"""
    result = RelationExtractService().recompute_from_review(payload.review_id, payload.operator)
    if result.get("error"):
        raise HTTPException(status_code=409, detail=result["error"])
    return result


@router.post("/recompute/compare")
def compare_recompute(payload: ComparePayload) -> dict[str, Any]:
    """比较两次派单/重算的关系明细差异。"""
    return RelationExtractService().compare_recompute_runs(
        payload.review_id_a, payload.review_id_b)
