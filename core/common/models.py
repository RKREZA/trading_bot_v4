from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(String, primary_key=True) # execution_id
    signal_id = Column(String)
    symbol = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    entry_price = Column(Float)
    exit_price = Column(Float)
    volume = Column(Float)
    sl = Column(Float)
    tp = Column(Float)
    pnl = Column(Float, default=0.0)
    status = Column(String, default='OPEN') # OPEN, CLOSED, REJECTED
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime)
    metadata_json = Column(JSON)

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String, unique=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String) # INFO, WARNING, ERROR, CRITICAL
    category = Column(String) # STRATEGY, RISK, EXECUTION, SYSTEM
    message = Column(String)
    data = Column(JSON)

class SystemState(Base):
    __tablename__ = 'system_state'
    
    key = Column(String, primary_key=True)
    value = Column(JSON)
    updated_at = Column(DateTime, default=datetime.utcnow)
