from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class RestorationJob(Base):
    __tablename__ = "restoration_jobs"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True)
    input_path = Column(String)
    output_path = Column(String, nullable=True)
    status = Column(String, default="pending")
    progress = Column(Integer, default=0)
    config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class VideoMetadata(Base):
    __tablename__ = "video_metadata"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer)
    duration = Column(Integer)
    frame_count = Column(Integer)
    resolution = Column(String)
    codec = Column(String)
