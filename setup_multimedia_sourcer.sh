#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[ERROR] See /tmp/msetup.log for details."' ERR
LOG=/tmp/msetup.log; : > "$LOG"

REPO_URL="https://github.com/StanchPillow55/Ideation-Automator.git"
REPO_DIR="Ideation-Automator"
APP_DIR="${REPO_DIR}/multimedia-sourcer"
BRANCH="mvp-multimedia-sourcer"

echo "[*] Clone/switch branch..." | tee -a "$LOG"
if [ ! -d "$REPO_DIR/.git" ]; then git clone "$REPO_URL" "$REPO_DIR" >>"$LOG" 2>&1; fi
git -C "$REPO_DIR" fetch --all --prune >>"$LOG" 2>&1 || true
BASE_BRANCH=$(git -C "$REPO_DIR" symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo main)
git -C "$REPO_DIR" switch "$BASE_BRANCH" >>"$LOG" 2>&1 || git -C "$REPO_DIR" checkout "$BASE_BRANCH" >>"$LOG" 2>&1
git -C "$REPO_DIR" switch -c "$BRANCH" >>"$LOG" 2>&1 || git -C "$REPO_DIR" switch "$BRANCH" >>"$LOG" 2>&1

echo "[*] Create dirs..." | tee -a "$LOG"
mkdir -p "$APP_DIR/api/routers" >>"$LOG" 2>&1
mkdir -p "$APP_DIR/api/services" >>"$LOG" 2>&1
mkdir -p "$APP_DIR/api/samples" >>"$LOG" 2>&1
mkdir -p "$APP_DIR/api/tests" >>"$LOG" 2>&1
mkdir -p "$APP_DIR/infra/db-init" >>"$LOG" 2>&1
mkdir -p "$APP_DIR/ui/app/(builder)" >>"$LOG" 2>&1
mkdir -p "$APP_DIR/ui/app/jobs" >>"$LOG" 2>&1
mkdir -p "$APP_DIR/ui/app/sources" >>"$LOG" 2>&1
mkdir -p "$APP_DIR/ui/components/ui" >>"$LOG" 2>&1
mkdir -p "$APP_DIR/ui/styles" >>"$LOG" 2>&1
mkdir -p "$APP_DIR/scripts" >>"$LOG" 2>&1
mkdir -p "$APP_DIR/data/exports" >>"$LOG" 2>&1

cat > "$APP_DIR/docker-compose.yml" <<'EOF'
name: multimedia-sourcer
services:
  db:
    image: pgvector/pgvector:pg16
    environment: { POSTGRES_DB: msourcer, POSTGRES_USER: msource, POSTGRES_PASSWORD: msource }
    ports: ["5432:5432"]
    volumes: [ "pgdata:/var/lib/postgresql/data", "./infra/db-init:/docker-entrypoint-initdb.d:ro" ]
    healthcheck: { test: ["CMD-SHELL","pg_isready -U msource -d msourcer"], interval: 5s, timeout: 5s, retries: 20 }
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    healthcheck: { test: ["CMD","redis-cli","ping"], interval: 5s, timeout: 5s, retries: 20 }
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment: { MINIO_ROOT_USER: admin, MINIO_ROOT_PASSWORD: admin123 }
    ports: ["9000:9000","9001:9001"]
    volumes: [ "minio_data:/data" ]
    healthcheck: { test: ["CMD","curl","-f","http://localhost:9000/minio/health/ready"], interval: 5s, timeout: 5s, retries: 20 }
  minio-setup:
    image: minio/mc:latest
    depends_on: { minio: { condition: service_healthy } }
    entrypoint: >
      /bin/sh -c "
      mc alias set local http://minio:9000 admin admin123 &&
      mc mb --ignore-existing local/ms-exports &&
      mc anonymous set download local/ms-exports || true && echo ready
      "
    restart: "no"
  api:
    build: { context: ./api, dockerfile: Dockerfile }
    environment:
      DATABASE_URL: postgresql+psycopg://msource:msource@db:5432/msourcer
      REDIS_URL: redis://redis:6379/0
      MINIO_ENDPOINT: http://minio:9000
      MINIO_ACCESS_KEY: admin
      MINIO_SECRET_KEY: admin123
      MINIO_BUCKET: ms-exports
      EXPORTS_DIR: /data/exports
      PYTHONDONTWRITEBYTECODE: "1"
      PYTHONUNBUFFERED: "1"
      CORS_ALLOW_ORIGIN: "*"
    volumes: [ "./data:/data" ]
    depends_on:
      db: { condition: service_healthy }
      redis: { condition: service_healthy }
      minio: { condition: service_healthy }
      minio-setup: { condition: service_completed_successfully }
    ports: ["8000:8000"]
  worker:
    build: { context: ./api, dockerfile: Dockerfile }
    command: celery -A celery_app.celery worker -Ofair -Q default -l INFO
    environment:
      DATABASE_URL: postgresql+psycopg://msource:msource@db:5432/msourcer
      REDIS_URL: redis://redis:6379/0
      MINIO_ENDPOINT: http://minio:9000
      MINIO_ACCESS_KEY: admin
      MINIO_SECRET_KEY: admin123
      MINIO_BUCKET: ms-exports
      EXPORTS_DIR: /data/exports
      PYTHONDONTWRITEBYTECODE: "1"
      PYTHONUNBUFFERED: "1"
    volumes: [ "./data:/data" ]
    depends_on:
      api: { condition: service_started }
      db: { condition: service_healthy }
      redis: { condition: service_healthy }
  flower:
    image: mher/flower:latest
    command: ["flower","--port=5555","--broker=redis://redis:6379/0"]
    ports: ["5555:5555"]
    depends_on: { redis: { condition: service_healthy } }
  ui:
    build: { context: ./ui, dockerfile: Dockerfile }
    environment: { NEXT_PUBLIC_API_URL: http://api:8000 }
    depends_on: { api: { condition: service_started } }
    ports: ["3000:3000"]
volumes: { pgdata: {}, minio_data: {} }
EOF

cat > "$APP_DIR/infra/db-init/init.sql" <<'EOF'
CREATE EXTENSION IF NOT EXISTS vector;
EOF

cat > "$APP_DIR/api/Dockerfile" <<'EOF'
FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc libpq-dev curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt
COPY . /app
EXPOSE 8000
CMD ["bash","-lc","python -m uvicorn app:app --host 0.0.0.0 --port 8000"]
EOF

cat > "$APP_DIR/api/requirements.txt" <<'EOF'
fastapi==0.111.0
uvicorn[standard]==0.30.1
sqlalchemy==2.0.31
psycopg[binary]==3.2.1
pgvector==0.2.5
pydantic==2.8.2
python-multipart==0.0.9
celery[redis]==5.4.0
redis==5.0.7
requests==2.32.3
trafilatura==1.12.2
python-pptx==0.6.21
boto3==1.34.162
markdown==3.6
reportlab==4.2.2
orjson==3.10.7
black==24.8.0
pytest==8.3.2
httpx==0.27.0
EOF

cat > "$APP_DIR/api/app.py" <<'EOF'
import os; from fastapi import FastAPI; from fastapi.middleware.cors import CORSMiddleware; from fastapi.staticfiles import StaticFiles
from db import init_db; from routers import sources,pipelines,packs
app=FastAPI(title="Multimedia Sourcer API",version="0.1.0")
orig=os.getenv("CORS_ALLOW_ORIGIN","*"); app.add_middleware(CORSMiddleware,allow_origins=["*"] if orig=="*" else [orig],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
@app.on_event("startup")
def _startup(): init_db(); os.makedirs(os.getenv("EXPORTS_DIR","/data/exports"),exist_ok=True)
app.include_router(sources.router,prefix="/v1",tags=["sources"]); app.include_router(pipelines.router,prefix="/v1",tags=["pipelines"]); app.include_router(packs.router,prefix="/v1",tags=["packs"])
app.mount("/exports",StaticFiles(directory=os.getenv("EXPORTS_DIR","/data/exports")),name="exports")
@app.get("/health") 
def health(): return {"status":"ok"}
EOF

cat > "$APP_DIR/api/db.py" <<'EOF'
import os; from contextlib import contextmanager; from sqlalchemy import create_engine,text; from sqlalchemy.orm import sessionmaker,declarative_base
DATABASE_URL=os.getenv("DATABASE_URL","postgresql+psycopg://msource:msource@db:5432/msourcer"); engine=create_engine(DATABASE_URL,pool_pre_ping=True)
SessionLocal=sessionmaker(bind=engine,autoflush=False,autocommit=False); Base=declarative_base()
def init_db(): 
    with engine.connect() as c:
        try: c.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        except Exception: pass
    Base.metadata.create_all(bind=engine)
@contextmanager
def session_scope():
    s=SessionLocal()
    try: yield s; s.commit()
    except Exception: s.rollback(); raise
    finally: s.close()
EOF

cat > "$APP_DIR/api/models.py" <<'EOF'
import uuid,enum; from datetime import datetime
from sqlalchemy import Column,String,DateTime,Text,Integer,ForeignKey,Float,Enum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from db import Base; from pgvector.sqlalchemy import Vector
EMBED_DIM=8
class SourceType(str,enum.Enum): web="web"; youtube="youtube"; pdf="pdf"; docx="docx"; pptx="pptx"; audio="audio"; instagram="instagram"
def gen_id(p): return f"{p}_{uuid.uuid4().hex[:8]}"
class Document(Base):
    __tablename__="documents"; id=Column(String,primary_key=True,default=lambda:gen_id("doc"))
    source_type=Column(Enum(SourceType),nullable=False); url_or_path=Column(String); publisher=Column(String); author=Column(String); published_at=Column(String)
    language=Column(String,default="en"); raw_text=Column(Text,default=""); media=Column(JSONB,default=list); tables=Column(JSONB,default=list); created_at=Column(DateTime,default=datetime.utcnow)
    chunks=relationship("Chunk",back_populates="document",cascade="all, delete-orphan"); claims=relationship("Claim",back_populates="document",cascade="all, delete-orphan")
class Chunk(Base):
    __tablename__="chunks"; id=Column(String,primary_key=True,default=lambda:gen_id("chk"))
    doc_id=Column(String,ForeignKey("documents.id"),nullable=False); text=Column(Text,nullable=False); order=Column(Integer,default=0); embedding=Column(Vector(EMBED_DIM))
    document=relationship("Document",back_populates="chunks")
class Claim(Base):
    __tablename__="claims"; claim_id=Column(String,primary_key=True,default=lambda:gen_id("clm"))
    doc_id=Column(String,ForeignKey("documents.id"),nullable=False); chunk_id=Column(String,ForeignKey("chunks.id"))
    text=Column(Text,nullable=False); support_type=Column(String,default="quote"); support_span=Column(JSONB,default=dict)
    strength=Column(Float,default=0.75); topics=Column(JSONB,default=list); entities=Column(JSONB,default=list); citation_key=Column(String,nullable=False)
    document=relationship("Document",back_populates="claims")
class SlidePack(Base):
    __tablename__="slidepacks"; pack_id=Column(String,primary_key=True,default=lambda:gen_id("pack"))
    title=Column(String,default="Synthesis Pack"); thesis=Column(Text,default=""); sections=Column(JSONB,default=list); references=Column(JSONB,default=list); exports=Column(JSONB,default=dict); created_at=Column(DateTime,default=datetime.utcnow)
class Job(Base):
    __tablename__="jobs"; job_id=Column(String,primary_key=True,default=lambda:gen_id("job"))
    source_ids=Column(JSONB,default=list); status=Column(String,default="queued"); stages=Column(JSONB,default=dict); result=Column(JSONB,default=dict); error=Column(Text)
    created_at=Column(DateTime,default=datetime.utcnow); updated_at=Column(DateTime,default=datetime.utcnow,onupdate=datetime.utcnow)
EOF

cat > "$APP_DIR/api/schemas.py" <<'EOF'
from typing import List,Optional,Dict,Any; from pydantic import BaseModel,Field; from models import SourceType
class SourceCreateJson(BaseModel): source_type:SourceType; url_or_path:Optional[str]=None; language:Optional[str]="en"
class DocumentOut(BaseModel): id:str; source_type:SourceType; url_or_path:Optional[str]; language:Optional[str]; class Config: from_attributes=True
class PipelineRunRequest(BaseModel): source_ids:List[str]; export_options:Optional[Dict[str,Any]]=Field(default_factory=dict)
class JobOut(BaseModel): job_id:str; status:str; stages:Dict[str,Any]; result:Dict[str,Any]=Field(default_factory=dict); error:Optional[str]=None
EOF

cat > "$APP_DIR/api/celery_app.py" <<'EOF'
import os; from celery import Celery
redis=os.getenv("REDIS_URL","redis://redis:6379/0"); celery=Celery("multimedia_sourcer",broker=redis,backend=redis); celery.conf.result_expires=3600
EOF

cat > "$APP_DIR/api/services/ingestion.py" <<'EOF'
import os,re,requests,trafilatura; from typing import Tuple
def _read_local(p): 
    try: return open(p,"r",encoding="utf-8",errors="ignore").read()
    except Exception: return "Stub content (binary/unreadable)."
def fetch_web(url)->Tuple[str,str]:
    if url.startswith("file://"): 
        p=url[len("file://"):]; c=_read_local(p); t=trafilatura.extract(c) or c[:4000]; return (t or "No text extracted.","en")
    r=requests.get(url,timeout=15); r.raise_for_status(); e=trafilatura.extract(r.text) or r.text; return (e[:20000],"en")
def fetch_youtube_stub(url)->str:
    v="/app/samples/youtube_sample.vtt"
    if os.path.exists(v):
        t=_read_local(v); out=[]; 
        for line in t.splitlines():
            if re.match(r"^\\d{2}:\\d{2}:",line): continue
            if line.strip() and not line.startswith("WEBVTT"): out.append(line.strip())
        return " ".join(out)[:20000]
    return "Sample YouTube transcript text with timestamps [00:00-00:05]."
def fetch_file_stub(path,stype)->str: return f"Stub extracted text from {stype.upper()} file: {os.path.basename(path)}."
def ingest_by_type(source_type,url_or_path)->Tuple[str,str]:
    st=source_type.lower()
    if st=="web": return fetch_web(url_or_path)
    if st=="youtube": return fetch_youtube_stub(url_or_path),"en"
    if st in {"pdf","docx","pptx","audio","instagram"}: return fetch_file_stub(url_or_path or "",st),"en"
    return "Unsupported source type.","en"
EOF

cat > "$APP_DIR/api/services/rag.py" <<'EOF'
import hashlib; from typing import List,Tuple; EMBED_DIM=8
def _vec(t:str)->List[float]:
    h=hashlib.sha256(t.encode()).digest(); vals=[h[i]/255.0 for i in range(EMBED_DIM)]; s=sum(vals) or 1.0; return [v/s for v in vals]
def chunk_text(text:str,max_chars:int=300)->List[str]:
    chunks=[]; cur=""; 
    for line in text.strip().replace("\r","\n").split("\n"):
        if len(cur)+len(line)+1>max_chars and cur: chunks.append(cur.strip()); cur=""
        cur+=(" " if cur else "")+line.strip()
    if cur: chunks.append(cur.strip()); 
    return chunks[:20] or [text[:max_chars]]
def embed_chunks(chunks:List[str])->List[List[float]]: return [_vec(c) for c in chunks]
def nearest_k(q:str,items:List[Tuple[str,List[float]]],k:int=3): 
    v=_vec(q); dot=lambda a,b: sum(x*y for x,y in zip(a,b)); s=[(t,dot(v,x)) for t,x in items]; s.sort(key=lambda x:x[1],reverse=True); return s[:k]
EOF

cat > "$APP_DIR/api/services/export.py" <<'EOF'
import os,json,boto3; from typing import Dict,Any,List
from markdown import markdown as md_to_html; from pptx import Presentation; from reportlab.lib.pagesizes import letter; from reportlab.pdfgen import canvas
ED=os.getenv("EXPORTS_DIR","/data/exports"); ME=os.getenv("MINIO_ENDPOINT","http://minio:9000"); AK=os.getenv("MINIO_ACCESS_KEY","admin"); SK=os.getenv("MINIO_SECRET_KEY","admin123"); B=os.getenv("MINIO_BUCKET","ms-exports")
def _ensure(): os.makedirs(ED,exist_ok=True)
def save_notes_md(pid,lines): _ensure(); p=os.path.join(ED,f"notes-{pid}.md"); open(p,"w",encoding="utf-8").write("\n".join(lines)); return p
def save_datapack_json(pid,data): _ensure(); p=os.path.join(ED,f"datapack-{pid}.json"); open(p,"w",encoding="utf-8").write(json.dumps(data,indent=2)); return p
def save_marp_markdown(pid,title,secs):
    _ensure(); p=os.path.join(ED,f"slides-{pid}.md"); L=["---","marp: true","theme: default","paginate: true","---",f"# {title}",""]
    for s in secs:
        L.append(f"## {s.get('title','Section')}"); [L.append(f"- {b.get('text','')} ({b.get('citation_key','')})") for b in s.get('bullets',[])]; L.append("")
    open(p,"w",encoding="utf-8").write("\n".join(L)); return p
def render_html_from_markdown(md_path,pid):
    _ensure(); html=f"<!doctype html><html><head><meta charset='utf-8'><title>Pack {pid}</title></head><body>{md_to_html(open(md_path,'r',encoding='utf-8').read())}</body></html>"
    out=os.path.join(ED,f"pack-{pid}.html"); open(out,"w",encoding="utf-8").write(html); return out
def render_pdf_simple(pid,title,secs):
    _ensure(); out=os.path.join(ED,f"pack-{pid}.pdf"); c=canvas.Canvas(out,pagesize=letter); w,h=letter; y=h-72; c.setFont("Helvetica-Bold",16); c.drawString(72,y,title); y-=24; c.setFont("Helvetica",11)
    for s in secs:
        c.drawString(72,y,f"Section: {s.get('title','')}"); y-=18
        for b in s.get("bullets",[]):
            c.drawString(84,y,(f"- {b.get('text','')} ({b.get('citation_key','')})")[:100]); y-=14
            if y<100: c.showPage(); y=h-72
    c.showPage(); c.save(); return out
def render_pptx(pid,title,secs):
    _ensure(); from pptx.util import Inches
    prs=Presentation(); s=prs.slides.add_slide(prs.slide_layouts[0]); s.shapes.title.text=title; s.placeholders[1].text="Generated by Multimedia Sourcer"
    for sec in secs:
        sl=prs.slides.add_slide(prs.slide_layouts[1]); sl.shapes.title.text=sec.get("title","Section"); tf=sl.placeholders[1].text_frame; tf.clear()
        for b in sec.get("bullets",[]): p=tf.add_paragraph(); p.text=f"{b.get('text','')} ({b.get('citation_key','')})"; p.level=0
    out=os.path.join(ED,f"pack-{pid}.pptx"); prs.save(out); return out
def upload_to_minio(local_path):
    s3=boto3.client("s3",endpoint_url=ME,aws_access_key_id=AK,aws_secret_access_key=SK,region_name="us-east-1"); k=os.path.basename(local_path); s3.upload_file(local_path,B,k); return f"s3://{B}/{k}"
EOF

cat > "$APP_DIR/api/routers/__init__.py" <<'EOF'
# package
EOF

cat > "$APP_DIR/api/routers/sources.py" <<'EOF'
import os; from typing import List,Optional; from fastapi import APIRouter,UploadFile,File,Form,HTTPException,Depends; from sqlalchemy.orm import Session
from db import SessionLocal; from models import Document,SourceType; from schemas import SourceCreateJson,DocumentOut
router=APIRouter()

def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/sources/json",response_model=DocumentOut)
def create_source_json(p:SourceCreateJson,db:Session=Depends(get_db)):
    d=Document(source_type=p.source_type,url_or_path=p.url_or_path,language=p.language or "en"); db.add(d); db.commit(); db.refresh(d); return d

@router.post("/sources",response_model=DocumentOut)
async def create_source_form(source_type:SourceType=Form(...),url:Optional[str]=Form(None),file:Optional[UploadFile]=File(None),db:Session=Depends(get_db)):
    path=url
    if file:
        os.makedirs("/data/uploads",exist_ok=True); dest=os.path.join("/data/uploads",file.filename)
        with open(dest,"wb") as f: f.write(await file.read()); path=dest
    if not path: raise HTTPException(status_code=400,detail="Provide url or file")
    d=Document(source_type=source_type,url_or_path=path,language="en"); db.add(d); db.commit(); db.refresh(d); return d

@router.get("/sources",response_model=List[DocumentOut])
def list_sources(db:Session=Depends(get_db)): return db.query(Document).order_by(Document.created_at.desc()).limit(100).all()
EOF

cat > "$APP_DIR/api/routers/pipelines.py" <<'EOF'
from fastapi import APIRouter,HTTPException,Depends; from sqlalchemy.orm import Session; from sqlalchemy import select
from db import SessionLocal; from schemas import PipelineRunRequest,JobOut; from models import Job,Document; from tasks import run_pipeline
router=APIRouter()

def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/pipelines/run",response_model=JobOut)
def run_pipeline_endpoint(req:PipelineRunRequest,db:Session=Depends(get_db)):
    got=db.execute(select(Document.id).where(Document.id.in_(req.source_ids))).scalars().all()
    if len(got)!=len(req.source_ids): raise HTTPException(status_code=400,detail="Some source_ids not found")
    job=Job(source_ids=req.source_ids,status="queued",stages={k:"pending" for k in ["ingest","normalize","embed","analyze","synthesize","export"]}); db.add(job); db.commit(); db.refresh(job)
    run_pipeline.delay(job.job_id,req.source_ids,req.export_options or {}); return JobOut(job_id=job.job_id,status=job.status,stages=job.stages,result=job.result,error=job.error)

@router.get("/jobs/{job_id}",response_model=JobOut)
def get_job(job_id:str,db:Session=Depends(get_db)):
    j=db.get(Job,job_id); 
    if not j: raise HTTPException(status_code=404,detail="Job not found")
    return JobOut(job_id=j.job_id,status=j.status,stages=j.stages,result=j.result,error=j.error)
EOF

cat > "$APP_DIR/api/routers/packs.py" <<'EOF'
from fastapi import APIRouter,Depends; from sqlalchemy.orm import Session; from db import SessionLocal; from models import SlidePack
router=APIRouter()

def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/packs")
def list_packs(db:Session=Depends(get_db)):
    ps=db.query(SlidePack).order_by(SlidePack.created_at.desc()).limit(50).all()
    return [{"pack_id":p.pack_id,"title":p.title,"thesis":p.thesis,"exports":p.exports,"created_at":p.created_at.isoformat()} for p in ps]
EOF

cat > "$APP_DIR/api/tasks.py" <<'EOF'
from sqlalchemy.orm import Session; from sqlalchemy import select
from celery_app import celery; from db import SessionLocal
from models import Job,Document,Chunk,Claim,SlidePack
from services import ingestion as ing, rag, export as ex

def _upd(s:Session,j:Job,st:str,stt:str): j.stages[st]=stt; s.add(j); s.commit()

def _docs(s:Session,ids): return s.execute(select(Document).where(Document.id.in_(ids))).scalars().all()

@celery.task(name="run_pipeline")
def run_pipeline(job_id:str,source_ids,export_options):
    s=SessionLocal(); j=s.get(Job,job_id)
    if not j: s.close(); return
    try:
        j.status="running"; s.add(j); s.commit()
        _upd(s,j,"ingest","running"); ds=_docs(s,source_ids)
        for d in ds: txt,lang=ing.ingest_by_type(d.source_type.value,d.url_or_path or ""); d.raw_text=txt; d.language=lang; s.add(d)
        s.commit(); _upd(s,j,"ingest","completed")
        _upd(s,j,"normalize","running"); 
        for d in ds:
            s.query(Chunk).filter(Chunk.doc_id==d.id).delete()
            for i,ch in enumerate(rag.chunk_text(d.raw_text or "")): s.add(Chunk(doc_id=d.id,text=ch,order=i))
        s.commit(); _upd(s,j,"normalize","completed")
        _upd(s,j,"embed","running")
        for d in ds:
            chs=s.execute(select(Chunk).where(Chunk.doc_id==d.id).order_by(Chunk.order)).scalars().all()
            vecs=rag.embed_chunks([c.text for c in chs])
            for c,v in zip(chs,vecs): c.embedding=v; s.add(c)
        s.commit(); _upd(s,j,"embed","completed")
        _upd(s,j,"analyze","running"); s.query(Claim).filter(Claim.doc_id.in_(source_ids)).delete(synchronize_session=False)
        for d in ds:
            chs=s.execute(select(Chunk).where(Chunk.doc_id==d.id).order_by(Chunk.order)).scalars().all()
            for c in chs[:5]:
                st="timestamp" if d.source_type.value=="youtube" else "quote"; span={"start":"00:00","end":"00:10"} if st=="timestamp" else {"charRange":[0,min(50,len(c.text))]}
                s.add(Claim(doc_id=d.id,chunk_id=c.id,text=(c.text[:180]+"...") if len(c.text)>180 else c.text,support_type=st,support_span=span,citation_key=f"{d.id}:{c.id}"))
        s.commit(); _upd(s,j,"analyze","completed")
        _upd(s,j,"synthesize","running"); secs=[]; refs=[]
        for d in ds:
            cls=s.execute(select(Claim).where(Claim.doc_id==d.id)).scalars().all()
            secs.append({"title":f"{d.source_type.value.upper()} {d.id[:6]}","bullets":[{"text":cl.text,"citation_key":cl.citation_key} for cl in cls[:5]],"figures":[]})
            refs.append({"doc_id":d.id,"source_type":d.source_type.value,"url_or_path":d.url_or_path})
        p=SlidePack(title="Multimedia Sourcer Pack",thesis="Synthesized claims and summaries.",sections=secs,references=refs,exports={}); s.add(p); s.commit(); s.refresh(p)
        _upd(s,j,"synthesize","completed")
        _upd(s,j,"export","running")
        notes=[f"- {d.id} includes timestamps like [00:00-00:10]." for d in ds if d.source_type.value=="youtube"] or ["No speaker notes available."]
        notes_p=ex.save_notes_md(p.pack_id,notes); md_p=ex.save_marp_markdown(p.pack_id,p.title,p.sections)
        html_p=ex.render_html_from_markdown(md_p,p.pack_id); pdf_p=ex.render_pdf_simple(p.pack_id,p.title,p.sections); pptx_p=ex.render_pptx(p.pack_id,p.title,p.sections)
        all_claims=[]; 
        for d in ds:
            for cl in s.execute(select(Claim).where(Claim.doc_id==d.id)).scalars().all():
                all_claims.append({"claim_id":cl.claim_id,"doc_id":cl.doc_id,"chunk_id":cl.chunk_id,"text":cl.text,"support_type":cl.support_type,"support_span":cl.support_span,"citation_key":cl.citation_key})
        json_p=ex.save_datapack_json(p.pack_id,{"pack_id":p.pack_id,"title":p.title,"sections":p.sections,"references":p.references,"claims":all_claims,"coverage_confidence":{"coverage":0.8,"confidence":0.7}})
        _=[ex.upload_to_minio(x) for x in [pptx_p,pdf_p,html_p,notes_p,json_p]]
        p.exports={"pptx":pptx_p,"pdf":pdf_p,"html":html_p,"md":notes_p,"json":json_p}; s.add(p); s.commit()
        j.status="completed"; j.result={"pack_id":p.pack_id,"exports":p.exports}; s.add(j); s.commit(); s.close()
    except Exception as e:
        j.status="failed"; j.error=str(e); s.add(j); s.commit(); s.close(); raise
EOF

cat > "$APP_DIR/api/samples/sample.html" <<'EOF'
<!doctype html><html><head><meta charset="utf-8"><title>Sample Web Page</title></head><body><h1>Sample</h1><p>Small HTML for ingestion tests.</p><p>Another paragraph for chunking & claims.</p></body></html>
EOF

cat > "$APP_DIR/api/samples/youtube_sample.vtt" <<'EOF'
WEBVTT

00:00:00.000 --> 00:00:05.000
Welcome to the sample video. This caption demonstrates timestamps.

00:00:05.000 --> 00:00:10.000
We will extract claims from captions for testing purposes.
EOF

cat > "$APP_DIR/api/samples/sample.pdf" <<'EOF'
%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 62 >> stream
BT /F1 18 Tf 36 108 Td (Hello PDF from Multimedia Sourcer stub) Tj ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000010 00000 n 
0000000061 00000 n 
0000000115 00000 n 
0000000298 00000 n 
0000000412 00000 n 
trailer << /Root 1 0 R /Size 6 >>
startxref
501
%%EOF
EOF

cat > "$APP_DIR/api/tests/test_api.py" <<'EOF'
from fastapi.testclient import TestClient; from app import app; from db import init_db; from tasks import run_pipeline; import os
c=TestClient(app)
def test_pipeline():
    init_db()
    web=c.post("/v1/sources/json",json={"source_type":"web","url_or_path":"file:///app/samples/sample.html"}).json()["id"]
    pdf=c.post("/v1/sources/json",json={"source_type":"pdf","url_or_path":"/app/samples/sample.pdf"}).json()["id"]
    yt=c.post("/v1/sources/json",json={"source_type":"youtube","url_or_path":"https://youtube.com/watch?v=abc"}).json()["id"]
    job=c.post("/v1/pipelines/run",json={"source_ids":[web,pdf,yt],"export_options":{}}).json()["job_id"]
    run_pipeline(job,[web,pdf,yt],{})
    j=c.get(f"/v1/jobs/{job}").json(); assert j["status"]=="completed"; ex=j["result"]["exports"]
    assert all(os.path.exists(ex[k]) for k in ["pptx","pdf","html","md","json"])
EOF

cat > "$APP_DIR/ui/Dockerfile" <<'EOF'
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci || npm install
COPY . /app
RUN npm run build
EXPOSE 3000
CMD ["npm","start"]
EOF

cat > "$APP_DIR/ui/package.json" <<'EOF'
{"name":"multimedia-sourcer-ui","private":true,"version":"0.1.0","scripts":{"dev":"next dev -p 3000","build":"next build","start":"next start -p 3000","lint":"echo \"skip\""},"dependencies":{"next":"14.2.5","react":"18.3.1","react-dom":"18.3.1","tailwindcss":"3.4.10","autoprefixer":"10.4.20","postcss":"8.4.41"}}
EOF

cat > "$APP_DIR/ui/next.config.js" <<'EOF'
module.exports={experimental:{appDir:true}};
EOF

cat > "$APP_DIR/ui/postcss.config.js" <<'EOF'
module.exports={plugins:{tailwindcss:{},autoprefixer:{}}};
EOF

cat > "$APP_DIR/ui/tailwind.config.ts" <<'EOF'
import type { Config } from 'tailwindcss'; const config:Config={content:["./app/**/*.{ts,tsx}","./components/**/*.{ts,tsx}"],theme:{extend:{}},plugins:[]}; export default config;
EOF

cat > "$APP_DIR/ui/tsconfig.json" <<'EOF'
{"compilerOptions":{"target":"ES2022","lib":["dom","es2022"],"jsx":"preserve","module":"esnext","moduleResolution":"bundler","strict":true,"types":["react","react-dom","node"]}}
EOF

cat > "$APP_DIR/ui/styles/globals.css" <<'EOF'
@tailwind base; @tailwind components; @tailwind utilities; .container{max-width:64rem;margin:0 auto;padding:1.5rem}.card{border:1px solid #e5e7eb;border-radius:.5rem;padding:1rem;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04)}.btn{padding:.5rem 1rem;border-radius:.5rem;background:#111;color:#fff}.input{border:1px solid #e5e7eb;border-radius:.5rem;padding:.5rem .75rem;width:100%}
EOF

cat > "$APP_DIR/ui/components/ui/button.tsx" <<'EOF'
import * as React from "react"; export function Button(p:React.ButtonHTMLAttributes<HTMLButtonElement>){const{className="",...r}=p;return <button className={`btn ${className}`} {...r}/>;}
EOF

cat > "$APP_DIR/ui/components/ui/input.tsx" <<'EOF'
import * as React from "react"; export const Input=React.forwardRef<HTMLInputElement,React.InputHTMLAttributes<HTMLInputElement>>(({className="",...r},ref)=>(<input ref={ref} className={`input ${className}`} {...r}/>)); Input.displayName="Input";
EOF

cat > "$APP_DIR/ui/app/layout.tsx" <<'EOF'
import './globals.css'; import React from 'react'; export const metadata={title:'Multimedia Sourcer',description:'Ingest, synthesize, export'}; export default function RootLayout({children}:{children:React.ReactNode}){return(<html lang="en"><body><nav className="border-b"><div className="container flex gap-4"><a href="/" className="font-semibold">Multimedia Sourcer</a><a href="/sources" className="hover:underline">Sources</a><a href="/jobs" className="hover:underline">Jobs</a><a href="/builder" className="hover:underline">Builder/Preview</a></div></nav><main className="container">{children}</main></body></html>);} 
EOF

cat > "$APP_DIR/ui/app/page.tsx" <<'EOF'
export default function Home(){return(<div className="card"><h1 className="text-2xl font-semibold mb-2">Multimedia Sourcer</h1><p>Add sources, run pipeline, download artifacts.</p></div>);} 
EOF

cat > "$APP_DIR/ui/app/sources/page.tsx" <<'EOF'
'use client'; import React,{useEffect,useState} from 'react'; import {Button} from '../../components/ui/button'; import {Input} from '../../components/ui/input';
const API=process.env.NEXT_PUBLIC_API_URL||'http://localhost:8000'; type Doc={id:string;source_type:string;url_or_path?:string};
export default function Sources(){const[t,setT]=useState('web');const[u,setU]=useState('');const[f,setF]=useState<File|null>(null);const[D,setD]=useState<Doc[]>([]);
const load=async()=>{const r=await fetch(`${API}/v1/sources`);setD(await r.json());}; useEffect(()=>{load();},[]);
const submit=async(e:React.FormEvent)=>{e.preventDefault();const form=new FormData();form.append('source_type',t);if(f)form.append('file',f);if(u)form.append('url',u);const r=await fetch(`${API}/v1/sources`,{method:'POST',body:form});if(r.ok){setU('');setF(null);load();}else alert('Failed');};
return(<div className="card"><h2 className="text-xl font-semibold mb-4">Add Source</h2><form onSubmit={submit} className="space-y-3"><div><label className="block text-sm">Source Type</label><select className="input" value={t} onChange={e=>setT(e.target.value)}><option>web</option><option>youtube</option><option>pdf</option><option>docx</option><option>pptx</option><option>audio</option><option>instagram</option></select></div><div><label className="block text-sm">URL</label><Input value={u} onChange={e=>setU(e.target.value)} placeholder="https://example.com or file path" /></div><div><label className="block text-sm">Upload</label><input type="file" onChange={e=>setF(e.target.files?.[0]||null)} /></div><Button type="submit">Create Source</Button></form><h3 className="text-lg font-semibold mt-6">Recent Sources</h3><ul className="mt-2 space-y-1">{D.map(d=>(<li key={d.id} className="text-sm">{d.id} · {d.source_type} · {d.url_or_path||''}</li>))}</ul></div>);} 
EOF

cat > "$APP_DIR/ui/app/jobs/page.tsx" <<'EOF'
'use client'; import React,{useEffect,useState} from 'react'; import {Button} from '../../components/ui/button';
const API=process.env.NEXT_PUBLIC_API_URL||'http://localhost:8000'; type Doc={id:string;source_type:string}; type Job={job_id:string;status:string;stages:Record<string,string>;result?:any};
export default function Jobs(){const[D,setD]=useState<Doc[]>([]);const[S,setS]=useState<string[]>([]);const[J,setJ]=useState<Job|null>(null);
const load=async()=>{const r=await fetch(`${API}/v1/sources`);setD(await r.json());}; useEffect(()=>{load();},[]);
const run=async()=>{const r=await fetch(`${API}/v1/pipelines/run`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source_ids:S,export_options:{}})});setJ(await r.json());};
useEffect(()=>{if(!J)return;const t=setInterval(async()=>{const r=await fetch(`${API}/v1/jobs/${J.job_id}`);const j=await r.json();setJ(j);if(j.status==='completed'||j.status==='failed')clearInterval(t);},1500);return()=>clearInterval(t);},[J?.job_id]);
return(<div className="card"><h2 className="text-xl font-semibold mb-2">Run Pipeline</h2><div className="space-y-2"><div className="space-y-1">{D.map(d=>(<label key={d.id} className="flex items-center gap-2 text-sm"><input type="checkbox" checked={S.includes(d.id)} onChange={e=>setS(p=>e.target.checked?[...p,d.id]:p.filter(x=>x!==d.id))} />{d.id} · {d.source_type}</label>))}</div><Button onClick={run}>Start</Button></div>{J&&(<div className="mt-6"><h3 className="font-semibold">Job {J.job_id}</h3><div>Status: <span className="font-mono">{J.status}</span></div><ul className="mt-2 text-sm">{Object.entries(J.stages).map(([k,v])=>(<li key={k}>{k}: {v}</li>))}</ul>{J.status==='completed'&&(<div className="mt-3"><div className="font-semibold">Exports:</div><ul className="text-sm">{Object.entries(J.result?.exports||{}).map(([k,p])=>(<li key={k}><a className="text-blue-600 underline" href={`${API.replace(/\/$/,'')}/exports/${(p as string).split('/').pop()}`} target="_blank">{k}</a></li>))}</ul></div>)}</div>)}</div>);} 
EOF

cat > "$APP_DIR/ui/app/(builder)/page.tsx" <<'EOF'
'use client'; import React,{useEffect,useState} from 'react'; const API=process.env.NEXT_PUBLIC_API_URL||'http://localhost:8000';
export default function Builder(){const[P,setP]=useState<any[]>([]); useEffect(()=>{(async()=>{const r=await fetch(`${API}/v1/packs`);setP(await r.json());})()},[]); return(<div className="card"><h2 className="text-xl font-semibold mb-2">Builder / Preview</h2><ul className="space-y-2">{P.map(p=>(<li key={p.pack_id} className="border rounded p-3"><div className="font-semibold">{p.title}</div><div className="text-sm text-gray-600">{p.pack_id}</div><div className="mt-1">{Object.entries(p.exports||{}).map(([k,v])=>(<a key={k} className="text-blue-600 underline mr-3" href={`${API.replace(/\/$/,'')}/exports/${(v as string).split('/').pop()}`} target="_blank">{k}</a>))}</div></li>))}</ul></div>);} 
EOF

cat > "$APP_DIR/scripts/e2e_smoke.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
API_BASE="http://localhost:8000"
echo "Waiting for API..."; for i in {1..60}; do curl -sf "$API_BASE/health" >/dev/null && break; sleep 2; done
WEB_ID=$(curl -sS -X POST -H "Content-Type: application/json" -d '{"source_type":"web","url_or_path":"file:///app/samples/sample.html"}' $API_BASE/v1/sources/json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
PDF_ID=$(curl -sS -F "source_type=pdf" -F "file=@api/samples/sample.pdf" $API_BASE/v1/sources | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
YT_ID=$(curl -sS -X POST -H "Content-Type: application/json" -d '{"source_type":"youtube","url_or_path":"https://youtube.com/watch?v=abc"}' $API_BASE/v1/sources/json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
JOB_ID=$(curl -sS -X POST -H "Content-Type: application/json" -d "{\"source_ids\":[\"$WEB_ID\",\"$PDF_ID\",\"$YT_ID\"]}" $API_BASE/v1/pipelines/run | python3 -c 'import sys,json; print(json.load(sys.stdin)["job_id"])')
for i in {1..120}; do J=$(curl -sS $API_BASE/v1/jobs/$JOB_ID); S=$(echo "$J" | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])'); echo "Status: $S"; [ "$S" = "completed" ] && break; [ "$S" = "failed" ] && { echo "$J"; exit 1; }; sleep 2; done
ls -1 data/exports || true
EOF
chmod +x "$APP_DIR/scripts/e2e_smoke.sh"

cat > "$APP_DIR/Makefile" <<'EOF'
SHELL:=/bin/bash
up: ; docker compose up --build -d
down: ; docker compose down -v
logs: ; docker compose logs -f --tail=200
test: ; docker compose exec api pytest -q
seed: ; bash scripts/e2e_smoke.sh
fmt: ; docker compose exec api bash -lc 'black . || true'
EOF

cat > "$APP_DIR/.env.example" <<'EOF'
POSTGRES_DB=msourcer
POSTGRES_USER=msource
POSTGRES_PASSWORD=msource
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=admin123
MINIO_BUCKET=ms-exports
DATABASE_URL=postgresql+psycopg://msource:msource@db:5432/msourcer
REDIS_URL=redis://redis:6379/0
EXPORTS_DIR=/data/exports
CORS_ALLOW_ORIGIN=*
EOF

cat > "$APP_DIR/README.md" <<'EOF'
# Multimedia Sourcer (MVP)
FastAPI+Celery+Postgres/pgvector+Redis+MinIO+Next.js via Docker Compose. Start: docker compose up --build -d; seed: bash scripts/e2e_smoke.sh; tests: docker compose exec api pytest -q; artifacts: ./data/exports & http://localhost:8000/exports/
EOF

echo "[*] Files written to $APP_DIR" | tee -a "$LOG"
echo "[*] Bootstrap complete." | tee -a "$LOG"

