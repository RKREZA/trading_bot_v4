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


class AccountUpdate(BaseModel):
    password: Optional[str] = None
    is_active: Optional[bool] = None
    broker_utc_offset: Optional[int] = None
    label: Optional[str] = None


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
