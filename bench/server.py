"""Reliable talu server lifecycle management for benchmarking.

Guarantees:
- Refuses to start if an existing talu serve process or port listener exists.
- Only ever manages the process it started (never kills foreign processes).
- Health-checked readiness before returning from start().
- Graceful SIGINT shutdown with SIGKILL fallback for its own process.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from pathlib import Path

# Default binary path (relative to repo root).
_DEFAULT_BINARY = Path(__file__).resolve().parent.parent / "zig-out" / "bin" / "talu"

# Timeouts (seconds).
_START_TIMEOUT = 30
_STOP_TIMEOUT = 10
_POLL_INTERVAL = 0.15


class ServerError(RuntimeError):
    """Raised when server lifecycle operations fail."""


class TaluServer:
    """Manages a single talu serve process."""

    def __init__(
        self,
        *,
        binary: str | Path = _DEFAULT_BINARY,
        host: str = "127.0.0.1",
        port: int = 8258,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.binary = Path(binary)
        self.host = host
        self.port = port
        self.extra_args = extra_args or []
        self.env = env or {}
        self._proc: subprocess.Popen | None = None  # type: ignore[type-arg]

    # -- Public API ----------------------------------------------------------

    def start(self, *, timeout: float = _START_TIMEOUT) -> None:
        """Check for conflicts, launch server, wait for /health.

        Raises ServerError if another talu serve process or port listener
        already exists.  Never kills foreign processes.
        """
        self._assert_no_conflicts()

        if not self.binary.exists():
            raise ServerError(f"binary not found: {self.binary}")

        cmd = [
            str(self.binary),
            "serve",
            "--host", self.host,
            "--port", str(self.port),
            "--no-bucket",
            *self.extra_args,
        ]

        # Inherit current env, overlay config env vars.
        proc_env = os.environ.copy()
        proc_env.update(self.env)

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=proc_env,
            # Own process group so we can signal just our server.
            preexec_fn=os.setsid,
        )

        try:
            self._wait_healthy(timeout=timeout)
        except Exception:
            # Startup failed — clean up our own process.
            self.stop()
            raise

    def stop(self, *, timeout: float = _STOP_TIMEOUT) -> None:
        """Stop the server we started. Graceful SIGINT, then SIGKILL."""
        proc = self._proc
        self._proc = None

        if proc is None:
            return

        if proc.poll() is not None:
            # Already exited — just reap.
            proc.wait()
            return

        # Graceful: SIGINT to process group.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except ProcessLookupError:
            pass

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Force kill the entire process group.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)

        # Wait for port to be released by the OS.
        self._wait_port_free(timeout=5)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> TaluServer:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- Internals ------------------------------------------------------------

    def _assert_no_conflicts(self) -> None:
        """Refuse to start if a talu serve process or port listener exists."""
        # Check for existing talu serve processes.
        existing = self._find_talu_serve_pids()
        if existing:
            pids_str = ", ".join(str(p) for p in existing)
            raise ServerError(
                f"existing talu serve process(es) found: pid {pids_str}\n"
                f"Stop them before running benchmarks:\n"
                f"  kill -INT {pids_str}"
            )

        # Check for anything else on our port.
        if self._port_in_use():
            raise ServerError(
                f"port {self.port} is already in use by another process.\n"
                f"Find it with:  lsof -ti :{self.port}\n"
                f"Stop it with:  kill -INT $(lsof -ti :{self.port})"
            )

    def _find_talu_serve_pids(self) -> list[int]:
        """Return PIDs of running talu serve processes (excludes self and zombies).

        Matches processes where argv[0] ends with '/talu' (or is 'talu')
        and argv[1] is 'serve'.  This avoids false positives from unrelated
        processes that happen to have 'talu' in a path argument.
        """
        pids: list[int] = []
        my_pid = os.getpid()
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == my_pid:
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
            except (OSError, PermissionError):
                continue
            args = raw.decode(errors="replace").split("\0")
            # Need at least argv[0] and argv[1].
            if len(args) < 2:
                continue
            binary = args[0].rsplit("/", 1)[-1]  # basename
            if binary != "talu" or args[1] != "serve":
                continue
            if self._is_zombie(pid):
                continue
            pids.append(pid)
        return pids

    @staticmethod
    def _is_zombie(pid: int) -> bool:
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            for line in status.splitlines():
                if line.startswith("State:"):
                    return "Z" in line
        except OSError:
            pass
        return False

    def _wait_port_free(self, *, timeout: float) -> None:
        """Block until nothing is listening on our port."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._port_in_use():
                return
            time.sleep(_POLL_INTERVAL)
        raise ServerError(
            f"port {self.port} still in use after {timeout}s"
        )

    def _port_in_use(self) -> bool:
        """Check if anything is listening on host:port."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        try:
            sock.connect((self.host, self.port))
            sock.close()
            return True
        except (ConnectionRefusedError, OSError):
            return False

    def _wait_healthy(self, *, timeout: float) -> None:
        """Poll GET /health until 200 or timeout."""
        deadline = time.monotonic() + timeout
        last_err = ""
        while time.monotonic() < deadline:
            # Check if process died.
            if self._proc and self._proc.poll() is not None:
                stderr = ""
                if self._proc.stderr:
                    stderr = self._proc.stderr.read().decode(errors="replace")
                raise ServerError(
                    f"server exited with code {self._proc.returncode} "
                    f"during startup.\nstderr:\n{stderr}"
                )
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                sock.connect((self.host, self.port))
                sock.sendall(
                    b"GET /health HTTP/1.1\r\n"
                    b"Host: localhost\r\n"
                    b"Connection: close\r\n\r\n"
                )
                resp = sock.recv(4096)
                sock.close()
                if b"200" in resp and b"ok" in resp.lower():
                    return
                last_err = f"unexpected response: {resp[:200]}"
            except (ConnectionRefusedError, OSError) as e:
                last_err = str(e)
            time.sleep(_POLL_INTERVAL)

        raise ServerError(
            f"server not healthy after {timeout}s. Last error: {last_err}"
        )
