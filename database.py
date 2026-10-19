from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import json

Base      = declarative_base()
engine    = create_engine("sqlite:///self_rag.db")
DBSession = sessionmaker(bind=engine)


# ── Models ────────────────────────────────────────────────────────────────────

class QueryLog(Base):
    __tablename__ = "query_logs"
    id                  = Column(Integer, primary_key=True)
    query               = Column(String)
    reformulated_query  = Column(String,  nullable=True)
    answer              = Column(Text)
    retrieval_relevance = Column(Float)
    answer_grounding    = Column(Float)
    answer_quality      = Column(Float)
    average_score       = Column(Float)
    hallucination_score = Column(Float,   nullable=True)
    attempts            = Column(Integer)
    multi_doc_mode      = Column(Boolean, default=False)
    cited_sources       = Column(Text,    nullable=True)
    trace               = Column(Text,    nullable=True)
    session_id          = Column(Integer, nullable=True)      # saved_sessions.id (manual grouping)
    chat_session_id     = Column(String,  nullable=True)      # chat_sessions.id
    created_at          = Column(DateTime, default=datetime.utcnow)


class SavedSession(Base):
    __tablename__ = "saved_sessions"
    id          = Column(Integer, primary_key=True)
    name        = Column(String,  nullable=False)
    query_count = Column(Integer, default=0)
    created_at  = Column(DateTime, default=datetime.utcnow)


class ChatSession(Base):
    """Named chat session — groups ChatMessages into conversations."""
    __tablename__ = "chat_sessions"
    id         = Column(String,  primary_key=True)   # UUID string
    name       = Column(String,  default="New Chat")
    is_pinned  = Column(Integer, default=0)           # 0/1
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    """Persistent chat history — every user/assistant turn stored here."""
    __tablename__ = "chat_messages"
    id           = Column(Integer, primary_key=True)
    session_id   = Column(String,  nullable=True)     # FK to chat_sessions.id
    role         = Column(String)           # "user" | "assistant" | "fallback" | "greeting"
    content      = Column(Text)             # raw answer text
    msg_metadata = Column(Text, nullable=True)  # JSON: scores, hallucination, etc.
    created_at   = Column(DateTime, default=datetime.utcnow)


class Feedback(Base):
    """User thumbs up / thumbs down ratings on answers."""
    __tablename__ = "feedback"
    id         = Column(Integer, primary_key=True)
    query_id   = Column(Integer, nullable=True)
    query      = Column(String)
    answer     = Column(Text)
    rating     = Column(Integer)    # 1 = thumbs up, -1 = thumbs down
    comment    = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db():
    Base.metadata.create_all(engine)
    print("Database ready.")

def init_sessions():
    Base.metadata.create_all(engine)

def init_chat_history():
    Base.metadata.create_all(engine)


# ── Query logging ─────────────────────────────────────────────────────────────

def log_query(query, answer, retrieval_relevance, answer_grounding,
              answer_quality, average_score, attempts,
              reformulated_query=None, hallucination_score=None,
              multi_doc_mode=False, cited_sources=None, trace=None, session_id=None):
    db    = DBSession()
    entry = QueryLog(
        query               = query,
        reformulated_query  = reformulated_query,
        answer              = answer,
        retrieval_relevance = retrieval_relevance,
        answer_grounding    = answer_grounding,
        answer_quality      = answer_quality,
        average_score       = average_score,
        hallucination_score = hallucination_score,
        attempts            = attempts,
        multi_doc_mode      = multi_doc_mode,
        cited_sources       = json.dumps(cited_sources) if cited_sources else None,
        trace               = json.dumps(trace)         if trace         else None,
        chat_session_id     = session_id,
    )
    db.add(entry); db.commit()
    qid = entry.id
    db.close()
    return qid


def get_all_logs():
    db   = DBSession()
    logs = db.query(QueryLog).order_by(QueryLog.created_at.desc()).all()
    result = []
    for log in logs:
        result.append({
            "id"                 : log.id,
            "query"              : log.query,
            "reformulated_query" : log.reformulated_query,
            "answer"             : log.answer,
            "retrieval_relevance": log.retrieval_relevance,
            "answer_grounding"   : log.answer_grounding,
            "answer_quality"     : log.answer_quality,
            "average_score"      : log.average_score,
            "hallucination_score": log.hallucination_score,
            "attempts"           : log.attempts,
            "multi_doc_mode"     : log.multi_doc_mode,
            "cited_sources"      : json.loads(log.cited_sources) if log.cited_sources else [],
            "trace"              : json.loads(log.trace)         if log.trace         else [],
            "session_id"         : log.session_id,
            "created_at"         : log.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    db.close()
    return result


# ── Sessions ──────────────────────────────────────────────────────────────────

def create_session(name):
    db = DBSession(); s = SavedSession(name=name)
    db.add(s); db.commit(); sid = s.id; db.close(); return sid


def get_all_sessions():
    db   = DBSession()
    rows = db.query(SavedSession).order_by(SavedSession.created_at.desc()).all()
    result = [{"id": r.id, "name": r.name, "query_count": r.query_count,
               "created_at": r.created_at.strftime('%Y-%m-%d %H:%M:%S')} for r in rows]
    db.close(); return result


def assign_queries_to_session(query_ids, session_id):
    from sqlalchemy import text
    with engine.connect() as conn:
        for qid in query_ids:
            conn.execute(text("UPDATE query_logs SET session_id=:sid WHERE id=:qid"),
                         {"sid": session_id, "qid": qid})
        conn.execute(text("UPDATE saved_sessions SET query_count="
                          "(SELECT COUNT(*) FROM query_logs WHERE session_id=:sid) WHERE id=:sid"),
                     {"sid": session_id})
        conn.commit()


def get_session_queries(session_id):
    db   = DBSession()
    logs = db.query(QueryLog).filter_by(session_id=session_id)\
              .order_by(QueryLog.created_at.asc()).all()
    result = [{"id": l.id, "query": l.query, "answer": l.answer,
               "average_score": l.average_score, "hallucination_score": l.hallucination_score,
               "attempts": l.attempts, "reformulated_query": l.reformulated_query,
               "cited_sources": json.loads(l.cited_sources) if l.cited_sources else [],
               "created_at": l.created_at.strftime('%Y-%m-%d %H:%M:%S')} for l in logs]
    db.close(); return result


# ── Persistent chat history ───────────────────────────────────────────────────

def save_chat_message(role, content, metadata=None, session_id=None):
    """Saves a single chat message. session_id links it to a ChatSession."""
    db  = DBSession()
    msg = ChatMessage(
        session_id   = session_id,
        role         = role,
        content      = content,
        msg_metadata = json.dumps(metadata) if metadata else None
    )
    db.add(msg); db.commit()
    new_id = msg.id
    # bump session updated_at
    if session_id:
        sess = db.query(ChatSession).filter_by(id=session_id).first()
        if sess:
            sess.updated_at = datetime.utcnow()
            db.commit()
    db.close()
    return new_id


def get_chat_history(limit=100, session_id=None):
    """Returns last N messages for a session, oldest-first.
    Includes legacy messages (session_id IS NULL) so old chats
    still appear after upgrading to session-based storage."""
    db = DBSession()
    q  = db.query(ChatMessage)
    if session_id:
        q = q.filter(ChatMessage.session_id == session_id)
    # newest `limit` rows, returned oldest-first (old code returned the OLDEST `limit`)
    msgs = q.order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc()).limit(limit).all()
    msgs.reverse()
    result = [{
        "id"        : m.id,
        "role"      : m.role,
        "content"   : m.content,
        "metadata"  : json.loads(m.msg_metadata) if m.msg_metadata else None,
        "created_at": m.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        "session_id": m.session_id
    } for m in msgs]
    db.close()
    return result


def clear_chat_history(session_id=None):
    """Deletes chat messages — for a specific session or all."""
    db = DBSession()
    q  = db.query(ChatMessage)
    if session_id:
        q = q.filter(ChatMessage.session_id == session_id)
    q.delete()
    db.commit(); db.close()


# ── Chat Sessions ─────────────────────────────────────────────────────────────

def create_chat_session(name="New Chat", session_id=None):
    import uuid as _uuid
    db   = DBSession()
    sess = ChatSession(
        id         = session_id or str(_uuid.uuid4()),
        name       = name,
        is_pinned  = 0,
        created_at = datetime.utcnow(),
        updated_at = datetime.utcnow()
    )
    db.add(sess); db.commit()
    sid = sess.id; db.close()
    return sid


def get_all_chat_sessions():
    db   = DBSession()
    rows = db.query(ChatSession)\
             .order_by(ChatSession.is_pinned.desc(),
                       ChatSession.updated_at.desc()).all()
    result = [{
        "id"        : r.id,
        "name"      : r.name,
        "is_pinned" : bool(r.is_pinned),
        "created_at": r.created_at.strftime('%Y-%m-%d %H:%M'),
        "updated_at": r.updated_at.strftime('%Y-%m-%d %H:%M')
    } for r in rows]
    db.close()
    return result


def rename_chat_session(session_id, new_name):
    db   = DBSession()
    sess = db.query(ChatSession).filter_by(id=session_id).first()
    if sess:
        sess.name       = new_name
        sess.updated_at = datetime.utcnow()
        db.commit()
    db.close()


def toggle_pin_session(session_id):
    db   = DBSession()
    sess = db.query(ChatSession).filter_by(id=session_id).first()
    if sess:
        sess.is_pinned  = 0 if sess.is_pinned else 1
        sess.updated_at = datetime.utcnow()
        db.commit()
    db.close()


def delete_chat_session(session_id):
    db = DBSession()
    db.query(ChatMessage).filter_by(session_id=session_id).delete()
    db.query(ChatSession).filter_by(id=session_id).delete()
    db.commit(); db.close()


def get_session_preview(session_id):
    """Returns first user message as preview text for the session list."""
    db  = DBSession()
    msg = db.query(ChatMessage)\
            .filter_by(session_id=session_id, role='user')\
            .order_by(ChatMessage.created_at.asc()).first()
    db.close()
    return msg.content[:60] if msg else "Empty chat"

def save_feedback(query, answer, rating, query_id=None, comment=None):
    db = DBSession()
    fb = Feedback(query=query, answer=answer, rating=rating,
                  query_id=query_id, comment=comment)
    db.add(fb); db.commit(); db.close()


def get_all_feedback():
    db   = DBSession()
    rows = db.query(Feedback).order_by(Feedback.created_at.desc()).all()
    result = [{
        "id"        : r.id,
        "query_id"  : r.query_id,
        "query"     : r.query,
        "answer"    : (r.answer or "")[:200],
        "rating"    : r.rating,
        "comment"   : r.comment,
        "created_at": r.created_at.strftime('%Y-%m-%d %H:%M:%S')
    } for r in rows]
    db.close(); return result


def get_feedback_stats():
    db      = DBSession()
    total   = db.query(Feedback).count()
    up      = db.query(Feedback).filter_by(rating=1).count()
    down    = db.query(Feedback).filter_by(rating=-1).count()
    db.close()
    return {"total": total, "thumbs_up": up, "thumbs_down": down,
            "satisfaction_pct": round(up / total * 100) if total else 0}


# ── Bookmarks ─────────────────────────────────────────────────────────────────

class Bookmark(Base):
    __tablename__ = "bookmarks"
    id         = Column(Integer, primary_key=True)
    query      = Column(String)
    answer     = Column(Text)
    doc_id     = Column(String, nullable=True)
    avg_score  = Column(Float,  nullable=True)
    note       = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def save_bookmark(query, answer, doc_id=None, avg_score=None, note=None):
    db = DBSession()
    bm = Bookmark(query=query, answer=answer, doc_id=doc_id,
                  avg_score=avg_score, note=note)
    db.add(bm); db.commit()
    bid = bm.id; db.close()
    return bid


def get_all_bookmarks():
    db   = DBSession()
    rows = db.query(Bookmark).order_by(Bookmark.created_at.desc()).all()
    result = [{"id": r.id, "query": r.query, "answer": r.answer,
               "doc_id": r.doc_id, "avg_score": r.avg_score,
               "note": r.note,
               "created_at": r.created_at.strftime('%Y-%m-%d %H:%M:%S')}
              for r in rows]
    db.close(); return result


def delete_bookmark(bookmark_id):
    db = DBSession()
    db.query(Bookmark).filter_by(id=bookmark_id).delete()
    db.commit(); db.close()


# ── Doc Tags ──────────────────────────────────────────────────────────────────

class DocTag(Base):
    __tablename__ = "doc_tags"
    id         = Column(Integer, primary_key=True)
    doc_id     = Column(String, nullable=False)
    tag        = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def add_doc_tag(doc_id, tag):
    db  = DBSession()
    # avoid duplicates
    existing = db.query(DocTag).filter_by(doc_id=doc_id, tag=tag).first()
    if not existing:
        dt = DocTag(doc_id=doc_id, tag=tag)
        db.add(dt); db.commit()
    db.close()


def get_doc_tags(doc_id=None):
    db = DBSession()
    q  = db.query(DocTag)
    if doc_id: q = q.filter_by(doc_id=doc_id)
    rows = q.order_by(DocTag.created_at.asc()).all()
    result = [{"id": r.id, "doc_id": r.doc_id, "tag": r.tag} for r in rows]
    db.close(); return result


def remove_doc_tag(tag_id):
    db = DBSession()
    db.query(DocTag).filter_by(id=tag_id).delete()
    db.commit(); db.close()


def get_all_tags():
    """Returns all unique tags across all docs."""
    db   = DBSession()
    rows = db.query(DocTag.tag).distinct().all()
    db.close()
    return [r[0] for r in rows]


# ── Migration helpers ─────────────────────────────────────────────────────────

def migrate_schema():
    """Adds columns introduced after the first release (SQLite has no IF NOT EXISTS for columns)."""
    from sqlalchemy import text
    with engine.connect() as conn:
        cols = [r[1] for r in conn.execute(text("PRAGMA table_info(query_logs)")).fetchall()]
        if cols and "chat_session_id" not in cols:
            conn.execute(text("ALTER TABLE query_logs ADD COLUMN chat_session_id VARCHAR"))
            conn.commit()


def adopt_orphan_messages():
    """
    Messages saved before sessions existed have session_id NULL. Attach them to one
    session so they stop leaking into every conversation. Returns the session id used
    (or None when there was nothing to adopt).
    """
    db = DBSession()
    orphans = db.query(ChatMessage).filter(ChatMessage.session_id == None).count()  # noqa: E711
    db.close()
    if not orphans:
        return None
    existing = get_all_chat_sessions()
    sid = existing[-1]["id"] if existing else create_chat_session(name="Earlier chat")
    db = DBSession()
    db.query(ChatMessage).filter(ChatMessage.session_id == None).update(     # noqa: E711
        {ChatMessage.session_id: sid})
    db.commit(); db.close()
    return sid


def get_logs_for_chat_session(session_id):
    db = DBSession()
    rows = db.query(QueryLog).filter_by(chat_session_id=session_id).all()
    out = [r.id for r in rows]
    db.close()
    return out


def get_query_id(query, answer=None):
    """Most recent query_logs.id for this query text (used to link feedback to a log row)."""
    db = DBSession()
    row = db.query(QueryLog).filter_by(query=query).order_by(QueryLog.id.desc()).first()
    db.close()
    return row.id if row else None



def delete_chat_turns(session_id, user_ids):
    """
    Deletes whole turns: each given USER message plus everything the system stored after it until the next
    user message (the answer, regenerated versions, not-found or fallback notes). Returns rows removed.
    """
    db = DBSession()
    removed = 0
    try:
        for uid in {int(x) for x in user_ids}:
            row = db.query(ChatMessage).filter_by(id=uid, session_id=session_id, role="user").first()
            if not row:
                continue
            nxt = (db.query(ChatMessage).filter(ChatMessage.session_id == session_id, ChatMessage.role == "user",
                                                ChatMessage.id > uid).order_by(ChatMessage.id.asc()).first())
            q = db.query(ChatMessage).filter(ChatMessage.session_id == session_id, ChatMessage.id >= uid)
            if nxt:
                q = q.filter(ChatMessage.id < nxt.id)
            removed += q.delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()
    return removed
