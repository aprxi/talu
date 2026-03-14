"""Verify server setup/teardown.

Run:  python bench/test_lifecycle.py
"""

from __future__ import annotations

import os
import signal
import socket
import sys
import time
from pathlib import Path

# Allow running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from server import TaluServer, ServerError


def port_in_use(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        sock.connect((host, port))
        sock.close()
        return True
    except (ConnectionRefusedError, OSError):
        return False


def assert_clean_state(label: str, host: str = "127.0.0.1", port: int = 8258) -> None:
    """Assert port is free."""
    assert not port_in_use(host, port), f"[{label}] port {port} still in use"


def test_start_stop() -> None:
    """Basic start -> health check -> stop -> verify clean."""
    print("test_start_stop ... ", end="", flush=True)
    srv = TaluServer()
    srv.start(timeout=30)
    assert srv.is_running(), "server should be running"
    assert srv.pid is not None
    srv.stop()
    assert not srv.is_running(), "server should not be running after stop"
    assert_clean_state("after stop")
    print("OK")


def test_context_manager() -> None:
    """Context manager guarantees teardown."""
    print("test_context_manager ... ", end="", flush=True)
    with TaluServer() as srv:
        assert srv.is_running()
    assert_clean_state("after context exit")
    print("OK")


def test_refuses_when_existing() -> None:
    """Starting a second server raises ServerError with actionable message."""
    print("test_refuses_when_existing ... ", end="", flush=True)
    srv1 = TaluServer()
    srv1.start(timeout=30)
    try:
        srv2 = TaluServer()
        try:
            srv2.start(timeout=5)
            srv2.stop()
            assert False, "should have raised ServerError"
        except ServerError as e:
            msg = str(e)
            assert "existing" in msg or "already in use" in msg, f"unclear error: {msg}"
            assert "kill" in msg.lower(), f"error should tell how to stop: {msg}"
    finally:
        srv1.stop()
    assert_clean_state("after refuses test")
    print("OK")


def test_repeated_cycles() -> None:
    """Start/stop 3 times in a row — no leaks."""
    print("test_repeated_cycles ... ", end="", flush=True)
    for i in range(3):
        with TaluServer() as srv:
            assert srv.is_running(), f"cycle {i}: not running"
        assert_clean_state(f"cycle {i}")
    print("OK")


def test_stop_idempotent() -> None:
    """Calling stop() multiple times is safe."""
    print("test_stop_idempotent ... ", end="", flush=True)
    srv = TaluServer()
    srv.start(timeout=30)
    srv.stop()
    srv.stop()  # second call should not raise
    srv.stop()  # third call should not raise
    assert_clean_state("after triple stop")
    print("OK")


def test_stop_after_crash() -> None:
    """If the server crashes, stop() still works and port is freed."""
    print("test_stop_after_crash ... ", end="", flush=True)
    srv = TaluServer()
    srv.start(timeout=30)
    pid = srv.pid
    # Simulate crash by killing the process directly.
    os.killpg(os.getpgid(pid), signal.SIGKILL)
    time.sleep(0.5)
    # stop() should handle the dead process gracefully.
    srv.stop()
    assert_clean_state("after crash stop")
    print("OK")


if __name__ == "__main__":
    binary = Path(__file__).resolve().parent.parent / "zig-out" / "bin" / "talu"
    if not binary.exists():
        print(f"ERROR: binary not found at {binary}", file=sys.stderr)
        print("Run: zig build release -Drelease", file=sys.stderr)
        sys.exit(1)

    # Ensure clean state before running tests.
    print("pre-check ... ", end="", flush=True)
    if port_in_use("127.0.0.1", 8258):
        print(f"FAIL: port 8258 in use. Stop existing server first.", file=sys.stderr)
        sys.exit(1)
    print("OK")

    tests = [
        test_start_stop,
        test_context_manager,
        test_refuses_when_existing,
        test_repeated_cycles,
        test_stop_idempotent,
        test_stop_after_crash,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
