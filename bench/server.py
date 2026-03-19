"""Reliable talu server lifecycle management for benchmarking.

Guarantees:
- Refuses to start only when the requested port is already in use.
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
        no_bucket: bool = True,
        bucket: str | Path | None = None,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.binary = Path(binary)
        self.host = host
        self.port = port
        self.no_bucket = no_bucket
        self.bucket = Path(bucket) if bucket is not None else None
        self.extra_args = extra_args or []
        self.env = env or {}
        self._proc: subprocess.Popen | None = None  # type: ignore[type-arg]
        self._socket_path: Path | None = None
        self._owns_socket_path: bool = False

        if self.no_bucket and self.bucket is not None:
            raise ValueError("no_bucket=True cannot be combined with bucket=...")

    # -- Public API ----------------------------------------------------------

    def start(self, *, timeout: float = _START_TIMEOUT) -> None:
        """Check for conflicts, launch server, wait for /health.

        Raises ServerError when the requested port is unavailable.
        Never kills foreign processes.
        """
        socket_path, owns_socket = self._resolve_socket_path()
        self._socket_path = socket_path
        self._owns_socket_path = owns_socket
        self._assert_no_conflicts()
        self._prepare_socket_path(socket_path)

        if not self.binary.exists():
            raise ServerError(f"binary not found: {self.binary}")

        cmd = [
            str(self.binary),
            "serve",
            "--host", self.host,
            "--port", str(self.port),
            "--socket", str(socket_path),
        ]
        if self.no_bucket:
            cmd.append("--no-bucket")
        elif self.bucket is not None:
            self.bucket.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--bucket", str(self.bucket)])
        cmd.extend(self.extra_args)

        # Inherit current env, overlay config env vars.
        proc_env = os.environ.copy()
        proc_env.update(self.env)

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
        self._cleanup_owned_socket_path()

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
        """Refuse to start only if the target port is already in use."""
        # Check for anything else on our port.
        if self._port_in_use():
            raise ServerError(
                f"port {self.port} is already in use by another process.\n"
                f"Find it with:  lsof -ti :{self.port}\n"
                f"Stop it with:  kill -INT $(lsof -ti :{self.port})"
            )

    def _resolve_socket_path(self) -> tuple[Path, bool]:
        """Resolve socket path from args, or create an isolated bench default."""
        explicit = self._extract_arg_value("--socket")
        if explicit:
            return Path(explicit).expanduser(), False
        return Path(f"/tmp/talu-bench-{self.port}.sock"), True

    def _extract_arg_value(self, key: str) -> str | None:
        """Extract a CLI argument value from ``extra_args``."""
        for i, arg in enumerate(self.extra_args):
            if arg == key and i + 1 < len(self.extra_args):
                return self.extra_args[i + 1]
            prefix = f"{key}="
            if arg.startswith(prefix):
                return arg[len(prefix):]
        return None

    def _prepare_socket_path(self, path: Path) -> None:
        """Remove stale socket files while refusing active socket collisions."""
        if not path.exists():
            return
        if self._socket_in_use(path):
            raise ServerError(f"socket {path} is already in use by another process.")
        try:
            path.unlink()
        except OSError as e:
            raise ServerError(f"failed to remove stale socket {path}: {e}") from e

    def _cleanup_owned_socket_path(self) -> None:
        """Best-effort cleanup for auto-managed socket paths."""
        path = self._socket_path
        owns = self._owns_socket_path
        self._socket_path = None
        self._owns_socket_path = False
        if not path or not owns:
            return
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass

    def _socket_in_use(self, path: Path) -> bool:
        """Check if a Unix domain socket path is accepting connections."""
        if not path.exists():
            return False
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(0.2)
        try:
            sock.connect(str(path))
            return True
        except OSError:
            return False
        finally:
            sock.close()

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
                raise ServerError(
                    f"server exited with code {self._proc.returncode} "
                    f"during startup."
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
