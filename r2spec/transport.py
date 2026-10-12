"""Bounded, synchronous static inspection through the local r2 NUL protocol."""
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import threading
import time
from typing import Any

from r2spec._child import ChildCleanup, attach_cleanup


class TransportError(RuntimeError):
    """A failed operation; code distinguishes input, transport and data failures."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.cleanup: ChildCleanup | None = None


def _reject_constant(value: str) -> Any:
    raise ValueError("non-JSON numeric constant")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


class R2Session:
    """One target and one outstanding operation, with no raw-command interface.

    read_bytes addresses physical file offsets, not virtual addresses. Cancellation
    is cooperative through an Event; transport failures permanently close a session.
    This is not a sandbox for the radare2 executable. Frames have no request IDs:
    rejecting queued bytes does not authenticate a schema-valid unsolicited frame
    arriving after a request is written.

    Incomplete cleanup is exposed by session.cleanup and exception.cleanup.
    Retain the handle and explicitly retry; do not serialize it or concurrently
    retry it while session cleanup is active. The host must not reap our child.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        executable: str | os.PathLike[str] = "r2",
        startup_timeout: float = 5.0,
        timeout: float = 2.0,
        max_response_bytes: int = 1024 * 1024,
        cancel: threading.Event | None = None,
    ):
        for value in (startup_timeout, timeout):
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0
            ):
                raise TransportError("input", "deadlines must be finite positive seconds")
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise TransportError("input", "response limit must be a positive integer")
        try:
            raw_path = os.fspath(path)
            if not isinstance(raw_path, str) or any(c in raw_path for c in "\0\r\n"):
                raise ValueError("invalid path")
            self._path = Path(raw_path).resolve(strict=True)
            if not self._path.is_file():
                raise ValueError("not a regular file")
            stat = self._path.stat()
        except (OSError, TypeError, ValueError) as exc:
            raise TransportError("input", "target must be an existing regular file") from exc

        program = shutil.which(os.fspath(executable))
        if program is None:
            raise TransportError("unavailable", "radare2 executable is unavailable")
        self._stamp = self._file_stamp(stat)
        self._size = stat.st_size
        self._timeout = timeout
        self._limit = max_response_bytes
        self._cancel = cancel
        self._lock = threading.Lock()
        self._close_request = threading.Event()
        self._process: subprocess.Popen | None = None
        self._selector: selectors.BaseSelector | None = None
        self._closed = True
        self._check_cancel()
        deadline = time.monotonic() + startup_timeout
        self._child = ChildCleanup()
        try:
            self._closed = False
            try:
                # The first store of a returned handle establishes ownership.
                self._child._process = subprocess.Popen(
                    [program, "-NN", "-q0", "-e", "scr.color=0",
                     "-e", "scr.interactive=false", "-e", "io.va=false", str(self._path)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, bufsize=0,
                    start_new_session=True, close_fds=True,
                )
            except OSError as exc:
                raise TransportError("unavailable", "cannot start radare2") from exc
            self._process = self._child._process
            os.set_blocking(self._process.stdin.fileno(), False)
            os.set_blocking(self._process.stdout.fileno(), False)
            self._selector = selectors.DefaultSelector()
            self._selector.register(self._process.stdout, selectors.EVENT_READ)
            if self._frame(b"", deadline, startup=True) != b"":
                raise TransportError("protocol", "unexpected startup frame")
            self._check_source()
        except BaseException as exc:
            failure = self._teardown()
            attach_cleanup(exc, self.cleanup)
            if failure is not None and self.cleanup is None:
                exc.add_note(str(failure))
            raise

    @staticmethod
    def _file_stamp(stat: os.stat_result) -> tuple[int, ...]:
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def cleanup(self) -> ChildCleanup | None:
        if self._child.state in ("pending", "lost"):
            return self._child
        return None

    def _check_process(self) -> None:
        try:
            status = self._child.observe()
        except OSError as exc:
            raise TransportError("process", "cannot observe radare2") from exc
        if status != "running":
            raise TransportError("process", "radare2 is no longer available")

    def _check_cancel(self) -> None:
        if self._close_request.is_set():
            raise TransportError("closed", "session closed during operation")
        if self._cancel is not None and self._cancel.is_set():
            raise TransportError("cancelled", "operation cancelled")

    def _check_source(self) -> None:
        try:
            current = self._file_stamp(self._path.stat())
        except OSError as exc:
            raise TransportError("source_changed", "target is no longer available") from exc
        if current != self._stamp:
            raise TransportError("source_changed", "target changed during the session")

    def _frame(self, command: bytes, deadline: float, *, startup: bool = False) -> bytes:
        pending = memoryview(command)
        output = bytearray()
        writing = bool(pending)
        if writing:
            self._selector.register(self._process.stdin, selectors.EVENT_WRITE)
        try:
            while True:
                self._check_cancel()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    code = "startup_timeout" if startup else "timeout"
                    raise TransportError(code, "radare2 deadline exceeded")
                self._check_process()
                for key, _ in self._selector.select(min(remaining, 0.05)):
                    if key.fileobj is self._process.stdin:
                        try:
                            sent = os.write(key.fd, pending)
                        except BlockingIOError:
                            continue
                        except OSError as exc:
                            raise TransportError("process", "radare2 input pipe failed") from exc
                        if not sent:
                            raise TransportError("process", "radare2 input pipe closed")
                        pending = pending[sent:]
                        if not pending:
                            self._selector.unregister(self._process.stdin)
                            writing = False
                        continue
                    if pending:
                        raise TransportError("protocol", "response arrived before request was sent")
                    try:
                        chunk = os.read(key.fd, min(65536, self._limit - len(output) + 1))
                    except BlockingIOError:
                        continue
                    except OSError as exc:
                        raise TransportError("process", "radare2 output pipe failed") from exc
                    if not chunk:
                        raise TransportError("process", "radare2 output pipe closed")
                    end = chunk.find(b"\0")
                    output.extend(chunk if end < 0 else chunk[:end])
                    if len(output) > self._limit:
                        raise TransportError("overflow", "radare2 response exceeds byte limit")
                    if end >= 0:
                        if chunk[end + 1:]:
                            raise TransportError("protocol", "unsolicited bytes after response frame")
                        return bytes(output)
        finally:
            if writing:
                self._selector.unregister(self._process.stdin)

    def _exchange(self, command: str, count: int | None = None) -> Any:
        if self._close_request.is_set():
            raise TransportError("closed", "session is closed")
        if not self._lock.acquire(blocking=False):
            if self._close_request.is_set():
                raise TransportError("closed", "session is closed")
            raise TransportError("busy", "another operation owns this session")
        try:
            if self._closed or self._close_request.is_set():
                raise TransportError("closed", "session is closed")
            try:
                self._check_source()
                try:
                    pending = os.read(self._process.stdout.fileno(), 1)
                except BlockingIOError:
                    pending = None
                except (OSError, ValueError) as exc:
                    raise TransportError("process", "radare2 output pipe failed") from exc
                self._check_process()
                if pending == b"":
                    raise TransportError("process", "radare2 is no longer available")
                if pending:
                    raise TransportError("protocol", "unsolicited output before request")
                raw = self._frame(
                    command.encode("ascii") + b"\n", time.monotonic() + self._timeout,
                )
                try:
                    result = json.loads(
                        raw.decode("utf-8"), parse_constant=_reject_constant,
                        object_pairs_hook=_unique_object,
                    )
                except (UnicodeError, ValueError, RecursionError) as exc:
                    raise TransportError("json", "response is not strict JSON") from exc
                self._check_source()
                self._check_cancel()
                if count is not None:
                    if (
                        not isinstance(result, list) or len(result) != count
                        or any(type(value) is not int or not 0 <= value <= 255 for value in result)
                    ):
                        raise TransportError("schema", "byte response has an invalid shape")
                    return bytes(result)
                core = result.get("core") if isinstance(result, dict) else None
                if (
                    not isinstance(core, dict)
                    or not isinstance(core.get("file"), str) or not core["file"]
                    or type(core.get("size")) is not int or core["size"] < 0
                    or ("bin" in result and not isinstance(result["bin"], dict))
                ):
                    raise TransportError("schema", "metadata response has an invalid shape")
                if core["size"] != self._size or core["file"] != str(self._path):
                    raise TransportError("schema", "metadata does not describe the opened file")
                return result
            except BaseException as exc:
                failure = self._teardown()
                attach_cleanup(exc, self.cleanup)
                if failure is not None and self.cleanup is None:
                    exc.add_note(str(failure))
                raise
        finally:
            self._lock.release()

    def read_bytes(self, offset: int, count: int) -> bytes:
        """Read 1..65536 bytes entirely within the original physical file."""
        if self._closed or self._close_request.is_set():
            raise TransportError("closed", "session is closed")
        if (
            type(offset) is not int or type(count) is not int
            or offset < 0 or offset >= 2**64 or not 1 <= count <= 65536
            or offset + count > self._size
        ):
            raise TransportError("input", "byte range is invalid or outside the original file")
        return self._exchange(f"pxj {count} @ {offset}", count)

    def metadata(self) -> dict[str, Any]:
        """Return r2 metadata after validating the required core fields."""
        return self._exchange("ij")

    def _teardown(self) -> TransportError | None:
        """The constructor or exchange-lock owner alone closes descriptors."""
        self._close_request.set()
        already_closed = self._closed
        self._closed = True
        failure = None
        resources = [self._selector]
        process = self._child._process
        if process is not None:
            resources.extend((process.stdin, process.stdout))
        for resource in resources:
            if resource is not None:
                try:
                    resource.close()
                except OSError as exc:
                    failure = TransportError("cleanup", f"descriptor cleanup failed: {exc}")
        state = self._child.retry() if already_closed else self._child.finish()
        if state != "complete":
            failure = TransportError("cleanup", self._child.detail)
            attach_cleanup(failure, self.cleanup)
        return failure

    def close(self) -> None:
        """Request shutdown and bound lock acquisition and owned cleanup.

        A cleanup error carries any pending handle. The caller must retain it
        and retry explicitly; kernel-stuck children cannot be reaped on demand.
        """
        self._close_request.set()
        if not self._lock.acquire(timeout=2.0):
            error = TransportError("cleanup", "active operation still owns session cleanup")
            attach_cleanup(error, self.cleanup)
            raise error
        try:
            failure = self._teardown()
            if failure is not None:
                raise failure
        finally:
            self._lock.release()

    def __enter__(self) -> "R2Session":
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> None:
        try:
            self.close()
        except TransportError as failure:
            if exc is None:
                raise
            attach_cleanup(exc, self.cleanup)
            if self.cleanup is None:
                exc.add_note(str(failure))
