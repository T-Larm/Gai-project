"""
In-memory session store for sentence-streamed /chat replies.

POST /chat_stream starts a worker thread that appends chunks to a session;
Unity polls GET /chat_stream/{id}?after=N. Finished sessions are evicted
lazily (on the next start()) after a TTL, so a crashed client can't leak
memory forever. Single-process only — matches the uvicorn deployment.
"""
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

SESSION_TTL_SECONDS = 300


class StreamSession:
    def __init__(self) -> None:
        self.chunks: List[Dict[str, Any]] = []
        self.done = False
        self.error: Optional[str] = None
        self.meta: Dict[str, Any] = {}
        self.created = time.time()
        self.lock = threading.Lock()


class StreamSessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, StreamSession] = {}
        self._lock = threading.Lock()

    def start(self, worker: Callable[[StreamSession], None]) -> str:
        self._evict_expired()
        session = StreamSession()
        session_id = uuid.uuid4().hex
        with self._lock:
            self._sessions[session_id] = session

        def run() -> None:
            try:
                worker(session)
            except Exception as exc:  # surfaced to the client via poll()
                with session.lock:
                    session.error = str(exc)
            finally:
                with session.lock:
                    session.done = True

        threading.Thread(target=run, daemon=True).start()
        return session_id

    def poll(self, session_id: str, after: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return None
        with session.lock:
            return {
                "chunks": [c for c in session.chunks if c["index"] > after],
                "done": session.done,
                "error": session.error,
                **session.meta,
            }

    def _evict_expired(self) -> None:
        now = time.time()
        with self._lock:
            expired = [
                sid for sid, s in self._sessions.items()
                if s.done and now - s.created > SESSION_TTL_SECONDS
            ]
            for sid in expired:
                del self._sessions[sid]
