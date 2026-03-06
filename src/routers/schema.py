from __future__ import annotations

import structlog
from fastapi import APIRouter, Request

from src.models import SchemaResponse

logger = structlog.get_logger()
router = APIRouter()


@router.get("/schema", response_model=SchemaResponse)
async def get_schema(request: Request) -> SchemaResponse:
    """Return the dynamically discovered graph schema."""
    schema_svc = request.app.state.schema_svc
    return SchemaResponse(
        labels=schema_svc.labels,
        relationship_types=schema_svc.relationship_types,
        label_count=len(schema_svc.labels),
        relationship_type_count=len(schema_svc.relationship_types),
        schema_text=schema_svc.get_full_schema_text(),
    )


@router.post("/schema/refresh")
async def refresh_schema(request: Request) -> dict:
    """Force re-discovery of schema, indexes, and concepts."""
    schema_svc = request.app.state.schema_svc
    index_svc = request.app.state.index_svc
    concept_svc = request.app.state.concept_svc

    await index_svc.discover()
    await schema_svc.discover()
    await concept_svc.discover()

    logger.info(
        "schema_refreshed",
        labels=len(schema_svc.labels),
        rels=len(schema_svc.relationship_types),
        indexes=index_svc.count,
        concepts=concept_svc.count,
    )

    return {
        "status": "refreshed",
        "labels": len(schema_svc.labels),
        "relationship_types": len(schema_svc.relationship_types),
        "indexes": index_svc.count,
        "concepts": concept_svc.count,
    }
