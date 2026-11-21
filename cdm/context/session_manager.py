"""Session Persistence and Ownership Subsystem for CDM-1.
Manages unique session creation, tenant/user isolation, and conversation context lifecycle.
"""

from typing import Dict, List, Optional, Any, Tuple
import time
import os
import json
import uuid
import sqlite3
import threading
from pydantic import BaseModel, Field


class ConversationSession(BaseModel):
    """Metadata describing an isolated conversation session."""
    session_id: str
    tenant_id: str
    user_id: str = "default_user"
    title: str = "New Conversation"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    turn_count: int = 0
    active_goals: List[str] = Field(default_factory=list)
    active_constraints: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionManager:
    """Thread-safe persistent session storage enforcing tenant and user isolation."""

    def __init__(self, db_path: str = "experiments/cognitive_memory.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS conversation_sessions (
                            session_id TEXT PRIMARY KEY,
                            tenant_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            title TEXT NOT NULL,
                            created_at REAL NOT NULL,
                            updated_at REAL NOT NULL,
                            turn_count INTEGER NOT NULL DEFAULT 0,
                            goals_json TEXT NOT NULL DEFAULT '[]',
                            constraints_json TEXT NOT NULL DEFAULT '[]',
                            metadata_json TEXT NOT NULL DEFAULT '{}'
                        );
                    """)
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_tenant ON conversation_sessions (tenant_id);")
            finally:
                conn.close()

    def create_session(
        self,
        tenant_id: str,
        user_id: str = "default_user",
        session_id: Optional[str] = None,
        title: str = "New Conversation",
        goals: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationSession:
        """Creates and stores a new unique session for a tenant."""
        s_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        now = time.time()
        goals_list = goals or []
        constraints_list = constraints or []
        meta_dict = metadata or {}

        session = ConversationSession(
            session_id=s_id,
            tenant_id=tenant_id,
            user_id=user_id,
            title=title,
            created_at=now,
            updated_at=now,
            turn_count=0,
            active_goals=goals_list,
            active_constraints=constraints_list,
            metadata=meta_dict,
        )

        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO conversation_sessions (
                            session_id, tenant_id, user_id, title,
                            created_at, updated_at, turn_count,
                            goals_json, constraints_json, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, (
                        s_id, tenant_id, user_id, title,
                        now, now, 0,
                        json.dumps(goals_list), json.dumps(constraints_list), json.dumps(meta_dict)
                    ))
            finally:
                conn.close()

        return session

    def get_or_create_session(
        self,
        tenant_id: str,
        session_id: Optional[str] = None,
        user_id: str = "default_user",
    ) -> ConversationSession:
        """Retrieves an existing session or provisions a new unique session."""
        if session_id and session_id != "default_session":
            existing = self.get_session(session_id=session_id, tenant_id=tenant_id)
            if existing:
                return existing
            return self.create_session(tenant_id=tenant_id, user_id=user_id, session_id=session_id)
        # Generate new unique session
        return self.create_session(tenant_id=tenant_id, user_id=user_id)

    def get_session(self, session_id: str, tenant_id: str) -> Optional[ConversationSession]:
        """Loads session metadata, enforcing tenant isolation."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("""
                    SELECT session_id, tenant_id, user_id, title,
                           created_at, updated_at, turn_count,
                           goals_json, constraints_json, metadata_json
                    FROM conversation_sessions
                    WHERE session_id = ? AND tenant_id = ?;
                """, (session_id, tenant_id))
                row = cur.fetchone()
                if not row:
                    return None
                data = dict(row)
                data["active_goals"] = json.loads(data["goals_json"])
                data["active_constraints"] = json.loads(data["constraints_json"])
                data["metadata"] = json.loads(data["metadata_json"])
                del data["goals_json"]
                del data["constraints_json"]
                del data["metadata_json"]
                return ConversationSession(**data)
            finally:
                conn.close()

    def list_sessions(self, tenant_id: str, limit: int = 50) -> List[ConversationSession]:
        """Lists active conversation sessions for a tenant."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("""
                    SELECT session_id, tenant_id, user_id, title,
                           created_at, updated_at, turn_count,
                           goals_json, constraints_json, metadata_json
                    FROM conversation_sessions
                    WHERE tenant_id = ?
                    ORDER BY updated_at DESC
                    LIMIT ?;
                """, (tenant_id, limit))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    data = dict(row)
                    data["active_goals"] = json.loads(data["goals_json"])
                    data["active_constraints"] = json.loads(data["constraints_json"])
                    data["metadata"] = json.loads(data["metadata_json"])
                    del data["goals_json"]
                    del data["constraints_json"]
                    del data["metadata_json"]
                    results.append(ConversationSession(**data))
                return results
            finally:
                conn.close()

    def update_session_activity(
        self,
        session_id: str,
        tenant_id: str,
        turn_increment: int = 1,
        goals: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
    ) -> None:
        """Updates last activity timestamp, turn count, and active constraints."""
        now = time.time()
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    # Fetch current turn count
                    cur = conn.execute("SELECT turn_count FROM conversation_sessions WHERE session_id = ? AND tenant_id = ?;", (session_id, tenant_id))
                    row = cur.fetchone()
                    current_turns = row[0] if row else 0
                    new_turns = current_turns + turn_increment

                    if goals is not None and constraints is not None:
                        conn.execute("""
                            UPDATE conversation_sessions
                            SET updated_at = ?, turn_count = ?, goals_json = ?, constraints_json = ?
                            WHERE session_id = ? AND tenant_id = ?;
                        """, (now, new_turns, json.dumps(goals), json.dumps(constraints), session_id, tenant_id))
                    else:
                        conn.execute("""
                            UPDATE conversation_sessions
                            SET updated_at = ?, turn_count = ?
                            WHERE session_id = ? AND tenant_id = ?;
                        """, (now, new_turns, session_id, tenant_id))
            finally:
                conn.close()

    def delete_session(self, session_id: str, tenant_id: str) -> bool:
        """Deletes session metadata, enforcing tenant isolation."""
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    cur = conn.execute("""
                        DELETE FROM conversation_sessions
                        WHERE session_id = ? AND tenant_id = ?;
                    """, (session_id, tenant_id))
                    return cur.rowcount > 0
            finally:
                conn.close()
