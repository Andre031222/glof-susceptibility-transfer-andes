import csv
import io
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import (Boolean, Column, DOUBLE_PRECISION as DoublePrecision, ForeignKey, Integer,
                        SmallInteger, String, Text, create_engine, func, select,
                        text, UniqueConstraint)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DB_URL = os.environ.get('GLOF_DB_URL', 'postgresql+psycopg2://glof:glof@localhost:5432/glof')
FRONTEND = Path(__file__).resolve().parent.parent / 'frontend'
THUMBS = Path(os.environ.get('GLOF_THUMBS', Path(__file__).resolve().parent / 'thumbs'))

engine = create_engine(DB_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine)
Base = declarative_base()


class Lake(Base):
    __tablename__ = 'lakes'
    lake_key = Column(Text, primary_key=True)
    idx = Column(Integer, unique=True)
    area_name = Column(Text, nullable=False)
    lat = Column(DoublePrecision, nullable=False)
    lon = Column(DoublePrecision, nullable=False)
    area_ha = Column(DoublePrecision)
    model_score = Column(DoublePrecision)
    dist_glacier_m = Column(DoublePrecision)
    elev_mean = Column(DoublePrecision)
    in_watchlist = Column(Boolean, default=False)
    known_glof = Column(Boolean, default=False)
    design_stratum = Column(Integer)
    inclusion_prob = Column(DoublePrecision)
    in_design_sample = Column(Boolean, default=False)
    thumb = Column(Text)


class Reviewer(Base):
    __tablename__ = 'reviewers'
    id = Column(Integer, primary_key=True)
    name = Column(Text, unique=True, nullable=False)


class Label(Base):
    __tablename__ = 'labels'
    id = Column(Integer, primary_key=True)
    lake_key = Column(Text, ForeignKey('lakes.lake_key', ondelete='CASCADE'), nullable=False)
    reviewer_id = Column(Integer, ForeignKey('reviewers.id', ondelete='CASCADE'), nullable=False)
    is_real = Column(Boolean)
    feature_type = Column(Text)
    confidence = Column(SmallInteger)
    outburst_evidence = Column(Boolean)
    note = Column(Text)
    __table_args__ = (UniqueConstraint('lake_key', 'reviewer_id'),)


Base.metadata.create_all(engine)
MIGRATIONS = [
    'ALTER TABLE lakes ADD COLUMN IF NOT EXISTS design_stratum INTEGER',
    'ALTER TABLE lakes ADD COLUMN IF NOT EXISTS inclusion_prob DOUBLE PRECISION',
    'ALTER TABLE lakes ADD COLUMN IF NOT EXISTS in_design_sample BOOLEAN DEFAULT FALSE',
    'ALTER TABLE labels ADD COLUMN IF NOT EXISTS outburst_evidence BOOLEAN',
]
if engine.dialect.name == 'postgresql':
    with engine.begin() as conn:
        for q in MIGRATIONS:
            conn.execute(text(q))

app = FastAPI(title='GLOF inventory validation')


def reviewer_id(s, name):
    name = (name or '').strip()[:60]
    if not name:
        raise HTTPException(400, 'reviewer name required')
    r = s.query(Reviewer).filter_by(name=name).one_or_none()
    if not r:
        r = Reviewer(name=name)
        s.add(r); s.commit()
    return r.id


class LabelIn(BaseModel):
    reviewer: str
    lake_key: str
    is_real: bool | None = None
    feature_type: str | None = None
    confidence: int | None = None
    outburst_evidence: bool | None = None
    note: str | None = None


@app.get('/api/lakes')
def lakes(reviewer: str, filter: str = 'all'):
    with Session() as s:
        rid = reviewer_id(s, reviewer)
        rows = s.query(Lake).order_by(Lake.area_name, Lake.in_watchlist.desc(),
                                      Lake.model_score.desc()).all()
        mine = {l.lake_key: l for l in s.query(Label).filter_by(reviewer_id=rid)}
        out = []
        for k in rows:
            off = (k.dist_glacier_m or 0) > 9830 or (k.elev_mean or 9999) < 2899
            if filter == 'watchlist' and not k.in_watchlist:
                continue
            if filter == 'off' and not off:
                continue
            if filter == 'design' and not (k.in_design_sample or k.in_watchlist or k.known_glof):
                continue
            m = mine.get(k.lake_key)
            if filter == 'todo' and m and m.is_real is not None:
                continue
            out.append({'lake_key': k.lake_key, 'idx': k.idx, 'area': k.area_name,
                        'lat': k.lat, 'lon': k.lon, 'area_ha': k.area_ha,
                        'score': k.model_score, 'dist': k.dist_glacier_m,
                        'elev': k.elev_mean, 'wl': k.in_watchlist, 'known': k.known_glof,
                        'thumb': k.thumb, 'off': off, 'design': bool(k.in_design_sample),
                        'mine': None if not m else {
                            'is_real': m.is_real, 'feature_type': m.feature_type,
                            'confidence': m.confidence,
                            'outburst_evidence': m.outburst_evidence, 'note': m.note}})
        return out


@app.post('/api/label')
def set_label(body: LabelIn):
    with Session() as s:
        rid = reviewer_id(s, body.reviewer)
        l = s.query(Label).filter_by(reviewer_id=rid, lake_key=body.lake_key).one_or_none()
        if not l:
            l = Label(reviewer_id=rid, lake_key=body.lake_key)
            s.add(l)
        l.is_real = body.is_real
        l.feature_type = body.feature_type
        l.confidence = body.confidence
        l.outburst_evidence = body.outburst_evidence
        l.note = body.note
        s.commit()
        return {'ok': True}


@app.get('/api/progress')
def progress(reviewer: str):
    with Session() as s:
        rid = reviewer_id(s, reviewer)
        total = s.query(func.count(Lake.lake_key)).scalar()
        wl = s.query(func.count(Lake.lake_key)).filter(Lake.in_watchlist).scalar()
        done = s.query(func.count(Label.id)).filter(Label.reviewer_id == rid,
                                                    Label.is_real.isnot(None)).scalar()
        wld = s.query(func.count(Label.id)).join(Lake, Lake.lake_key == Label.lake_key).filter(
            Label.reviewer_id == rid, Label.is_real.isnot(None), Lake.in_watchlist).scalar()
        return {'total': total, 'done': done, 'watchlist': wl, 'watchlist_done': wld}


@app.get('/api/stats')
def stats():
    with Session() as s:
        q = s.query(Lake.in_watchlist, Label.is_real).join(Label, Label.lake_key == Lake.lake_key)
        rows = q.filter(Label.is_real.isnot(None)).all()
        def rate(sub):
            n = len(sub); k = sum(1 for x in sub if x is False)
            return {'n': n, 'commission': k, 'rate_pct': round(100 * k / n, 1) if n else None}
        return {'overall': rate([r[1] for r in rows]),
                'watchlist': rate([r[1] for r in rows if r[0]]),
                'non_watchlist': rate([r[1] for r in rows if not r[0]])}


EXPORT_COLUMNS = ['lake_key', 'reviewer', 'area_name', 'lat', 'lon', 'area_ha', 'model_score',
                  'in_watchlist_top2pct', 'known_glof_source', 'design_stratum',
                  'inclusion_prob', 'in_design_sample', 'is_real_glacial_lake',
                  'feature_type', 'confidence', 'outburst_evidence', 'note']


def yn(v):
    return '' if v is None else ('Y' if v else 'N')


@app.get('/api/export')
def export():
    with Session() as s:
        rows = (s.query(Lake, Label, Reviewer)
                .outerjoin(Label, Label.lake_key == Lake.lake_key)
                .outerjoin(Reviewer, Reviewer.id == Label.reviewer_id)
                .order_by(Lake.idx, Reviewer.name).all())
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(EXPORT_COLUMNS)
        for k, l, r in rows:
            w.writerow([k.lake_key, r.name if r else '', k.area_name, k.lat, k.lon, k.area_ha,
                        k.model_score, int(bool(k.in_watchlist)), int(bool(k.known_glof)),
                        '' if k.design_stratum is None else k.design_stratum,
                        '' if k.inclusion_prob is None else repr(k.inclusion_prob),
                        int(bool(k.in_design_sample)),
                        yn(l.is_real) if l else '', (l.feature_type or '') if l else '',
                        '' if not l or l.confidence is None else l.confidence,
                        yn(l.outburst_evidence) if l else '', (l.note or '') if l else ''])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type='text/csv', headers={
        'Content-Disposition': 'attachment; filename=inventory_validation_results.csv'})


app.mount('/thumbs', StaticFiles(directory=str(THUMBS)), name='thumbs')


@app.get('/')
def index():
    return FileResponse(str(FRONTEND / 'index.html'))


app.mount('/', StaticFiles(directory=str(FRONTEND)), name='frontend')
