from fastapi import APIRouter, Depends, Query
from typing import Optional

from api.dependencies import get_db

router = APIRouter(prefix="/api/logs", tags=["logs"])


@router.get("/")
async def get_logs(
    limit: int = Query(default=100, le=500),
    offset: int = 0,
    category: Optional[str] = None,
    level: Optional[str] = None,
    db=Depends(get_db),
):
    from sqlalchemy import select, desc
    from core.common.models import AuditLog

    query = select(AuditLog).order_by(desc(AuditLog.timestamp)).offset(offset).limit(limit)

    if category:
        query = query.where(AuditLog.category == category)
    if level:
        query = query.where(AuditLog.level == level)

    result = await db.execute(query)
    logs = result.scalars().all()

    return {
        "count": len(logs),
        "offset": offset,
        "logs": [
            {
                "id": log.id,
                "event_id": log.event_id,
                "correlation_id": log.correlation_id,
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                "level": log.level,
                "category": log.category,
                "message": log.message,
                "data": log.data,
            }
            for log in logs
        ],
    }
