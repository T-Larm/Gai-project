"""StreamSessionManager: background worker, chunk polling, error capture."""
import threading
import time

from backend.streaming import StreamSessionManager


def _wait_done(manager, session_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = manager.poll(session_id, after=-1)
        if state is not None and state["done"]:
            return state
        time.sleep(0.01)
    raise AssertionError("session never finished")


def test_worker_chunks_become_pollable_and_done_is_set():
    manager = StreamSessionManager()

    def worker(session):
        for i in range(3):
            with session.lock:
                session.chunks.append({"index": i, "text": f"s{i}"})

    session_id = manager.start(worker)
    state = _wait_done(manager, session_id)

    assert [c["text"] for c in state["chunks"]] == ["s0", "s1", "s2"]
    assert state["error"] is None


def test_poll_after_filters_already_seen_chunks():
    manager = StreamSessionManager()

    def worker(session):
        for i in range(3):
            with session.lock:
                session.chunks.append({"index": i, "text": f"s{i}"})

    session_id = manager.start(worker)
    _wait_done(manager, session_id)

    state = manager.poll(session_id, after=1)
    assert [c["index"] for c in state["chunks"]] == [2]


def test_worker_exception_is_captured_not_raised():
    manager = StreamSessionManager()

    def worker(session):
        raise RuntimeError("XTTS exploded")

    session_id = manager.start(worker)
    state = _wait_done(manager, session_id)

    assert state["done"] is True
    assert "XTTS exploded" in state["error"]


def test_poll_unknown_session_returns_none():
    manager = StreamSessionManager()
    assert manager.poll("nope", after=-1) is None


def test_meta_is_included_in_poll_response():
    manager = StreamSessionManager()
    release = threading.Event()

    def worker(session):
        with session.lock:
            session.meta["guard"] = {"reason": "secret_low_trust"}
        release.wait(timeout=5)

    session_id = manager.start(worker)
    deadline = time.time() + 5
    state = manager.poll(session_id, after=-1)
    while not state.get("guard") and time.time() < deadline:
        time.sleep(0.01)
        state = manager.poll(session_id, after=-1)
    release.set()

    assert state["guard"] == {"reason": "secret_low_trust"}
    assert state["done"] is False
