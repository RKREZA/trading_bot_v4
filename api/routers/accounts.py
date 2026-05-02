from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from api.dependencies import get_db

router = APIRouter(prefix="/api/accounts", tags=["accounts"])


class AccountCreate(BaseModel):
    login: int
    password: str = ""
    server: str
    broker_utc_offset: int = 0
    label: str = ""
    max_drawdown_pct: float = 10.0
    risk_per_trade_pct: float = 1.0
    max_daily_drawdown_pct: float = 5.0


class AccountUpdate(BaseModel):
    password: Optional[str] = None
    is_active: Optional[bool] = None
    broker_utc_offset: Optional[int] = None
    label: Optional[str] = None
    max_drawdown_pct: Optional[float] = None
    risk_per_trade_pct: Optional[float] = None
    max_daily_drawdown_pct: Optional[float] = None


@router.get("/")
async def list_accounts(db=Depends(get_db)):
    from sqlalchemy import select
    from core.common.models import MTAccount

    result = await db.execute(select(MTAccount).order_by(MTAccount.id))
    accounts = result.scalars().all()
    return [
        {
            "id": a.id,
            "login": a.login,
            "server": a.server,
            "broker_utc_offset": a.broker_utc_offset,
            "is_active": a.is_active,
            "label": a.label,
            "has_password": bool(a.password),
            "max_drawdown_pct": a.max_drawdown_pct,
            "risk_per_trade_pct": a.risk_per_trade_pct,
            "max_daily_drawdown_pct": a.max_daily_drawdown_pct,
        }
        for a in accounts
    ]


@router.post("/")
async def create_account(body: AccountCreate, db=Depends(get_db)):
    from core.common.models import MTAccount

    account = MTAccount(
        login=body.login,
        password=body.password,
        server=body.server,
        broker_utc_offset=body.broker_utc_offset,
        label=body.label,
        max_drawdown_pct=body.max_drawdown_pct,
        risk_per_trade_pct=body.risk_per_trade_pct,
        max_daily_drawdown_pct=body.max_daily_drawdown_pct,
    )
    db.add(account)
    await db.commit()
    await db.refresh(account)
    return {"id": account.id, "login": account.login, "server": account.server}


@router.put("/{account_id}")
async def update_account(account_id: int, body: AccountUpdate, db=Depends(get_db)):
    from sqlalchemy import select
    from core.common.models import MTAccount

    result = await db.execute(select(MTAccount).where(MTAccount.id == account_id))
    account = result.scalar_one_or_none()
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    if body.password is not None:
        account.password = body.password
    if body.is_active is not None:
        account.is_active = body.is_active
    if body.broker_utc_offset is not None:
        account.broker_utc_offset = body.broker_utc_offset
    if body.label is not None:
        account.label = body.label
    if body.max_drawdown_pct is not None:
        account.max_drawdown_pct = body.max_drawdown_pct
    if body.risk_per_trade_pct is not None:
        account.risk_per_trade_pct = body.risk_per_trade_pct
    if body.max_daily_drawdown_pct is not None:
        account.max_daily_drawdown_pct = body.max_daily_drawdown_pct

    await db.commit()
    return {"status": "updated", "id": account_id}


@router.delete("/{account_id}")
async def delete_account(account_id: int, db=Depends(get_db)):
    from sqlalchemy import select
    from core.common.models import MTAccount

    result = await db.execute(select(MTAccount).where(MTAccount.id == account_id))
    account = result.scalar_one_or_none()
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    await db.delete(account)
    await db.commit()
    return {"status": "deleted", "id": account_id}
