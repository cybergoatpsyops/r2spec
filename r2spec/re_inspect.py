"""Reverse Engineering inspect-only run with private, all-or-nothing evidence.

Invoking a run is the operator's authorization to analyze the target. The target
is never executed. radare2 is not sandboxed and its replies are untagged, so
r2-derived facts stay "reported". Only metadata() and one bounded header read
are used, with no retry and no other r2 operation.

evidence.json exists only after a fully successful run. Any earlier failure,
including interruption, removes the files and directory this run created.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import tempfile

from r2spec.transport import R2Session, TransportError

SCHEMA = "r2spec.re.inspect/1"
EVIDENCE_NAME = "evidence.json"
HEADER_LIMIT = 64
_FACTS = (
    ("format", "core", str),
    ("bintype", "bin", str),
    ("class", "bin", str),
    ("arch", "bin", str),
    ("bits", "bin", int),
    ("endian", "bin", str),
    ("os", "bin", str),
    ("machine", "bin", str),
    ("lang", "bin", str),
    ("compiler", "bin", str),
    ("stripped", "bin", bool),
    ("static", "bin", bool),
    ("pic", "bin", bool),
    ("canary", "bin", bool),
    ("nx", "bin", bool),
    ("relro", "bin", str),
)


class InspectError(Exception):
    """A run that published nothing; kind selects the CLI exit status.

    Kinds: invocation, unavailable, failed, interrupted, cleanup_pending.
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class _Interrupts:
    """The first SIGINT raises KeyboardInterrupt; later ones cannot interrupt cleanup."""

    def __init__(self):
        self._previous = signal.getsignal(signal.SIGINT)
        self.ignoring = self._previous == signal.SIG_IGN
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame):
        if not self.ignoring:
            self.ignoring = True
            raise KeyboardInterrupt

    def restore(self) -> None:
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)


@contextlib.contextmanager
def _sigint_blocked():
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _stamp(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _check_paths(target: str, out: str) -> tuple[Path, Path]:
    message = "TARGET must be an existing regular file"
    try:
        resolved = Path(target).resolve(strict=True)
        regular = resolved.is_file()
    except (OSError, RuntimeError, ValueError) as exc:
        raise InspectError("invocation", message) from exc
    if not regular:
        raise InspectError("invocation", message)
    requested = Path(os.path.abspath(out))
    message = "the parent of --out must be an existing directory"
    try:
        parent = requested.parent.resolve(strict=True)
        directory = parent.is_dir()
    except (OSError, RuntimeError, ValueError) as exc:
        raise InspectError("invocation", message) from exc
    if not directory:
        raise InspectError("invocation", message)
    workspace = parent / requested.name
    if not requested.name or os.path.lexists(workspace):
        raise InspectError("invocation", "--out already exists; name a new directory")
    return resolved, workspace


def _observe(path: Path) -> tuple[os.stat_result, str, bytes]:
    """Hash the target in one pass and keep its header for corroboration."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise InspectError("invocation", "TARGET must be an existing regular file")
        digest = hashlib.sha256()
        header = b""
        total = 0
        while chunk := os.read(fd, 1 << 20):
            if len(header) < HEADER_LIMIT:
                header += chunk[:HEADER_LIMIT - len(header)]
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if _stamp(after) != _stamp(before) or total != before.st_size:
        raise InspectError("failed", "target changed while it was being hashed")
    return before, digest.hexdigest(), header


def _verify(resolved: Path, before: os.stat_result, ij: dict, header: bytes, remote: bytes) -> None:
    try:
        current = os.stat(resolved)
    except OSError as exc:
        raise InspectError("failed", "target is no longer available") from exc
    if _stamp(current) != _stamp(before):
        raise InspectError("failed", "target changed during inspection")
    core = ij["core"]
    if core["file"] != str(resolved) or core["size"] != before.st_size:
        raise InspectError("failed", "radare2 metadata does not describe the hashed target")
    if remote != header:
        raise InspectError("failed", "radare2 header bytes disagree with the local read")


def _render(resolved: Path, size: int, digest: str, header: bytes, ij: dict) -> bytes:
    facts = {}
    unknowns = [{"field": "tool.r2_version", "reason": "not_exposed"}]
    for name, section, kind in _FACTS:
        source = ij.get(section)
        if not isinstance(source, dict) or name not in source:
            reason = "absent"
        elif type(source[name]) is not kind:
            reason = "unexpected_type"
        elif source[name] == "":
            reason = "empty"
        elif name == "format" and source[name] == "null":
            reason = "unrecognized"
        else:
            facts[name] = {
                "value": source[name], "confidence": "reported", "source": f"ij.{section}.{name}",
            }
            continue
        unknowns.append({"field": f"facts.{name}", "reason": reason})
    operations = [{"op": "metadata", "r2": "ij"}]
    if header:
        operations.append({"op": "read_bytes", "offset": 0, "count": len(header)})
        facts["header_bytes"] = {
            "value": header.hex(), "offset": 0, "length": len(header),
            "confidence": "corroborated", "source": "pxj",
        }
    else:
        unknowns.append({"field": "facts.header_bytes", "reason": "empty_target"})
    document = {
        "schema": SCHEMA,
        "mode": "re",
        "operation": "inspect",
        "target": {"path": str(resolved), "size": size, "sha256": digest, "confidence": "observed"},
        "collection": {
            "transport": "r2spec.transport.R2Session",
            "operations": operations,
            "trust": {"r2_sandboxed": False, "replies_authenticated": False},
        },
        "facts": facts,
        "unknowns": sorted(unknowns, key=lambda item: item["field"]),
    }
    return (json.dumps(document, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rollback(final: Path | None, tmp: str | None, workspace: Path | None) -> str:
    """Remove only what this run created: files first, then the empty workspace."""
    problems = []
    for path, remove in ((final, os.unlink), (tmp, os.unlink), (workspace, os.rmdir)):
        if path is None:
            continue
        try:
            remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            problems.append(f"could not remove {path}: {exc.strerror}")
    return "; ".join(problems)


def _failure(exc: BaseException, handle, rollback: str) -> InspectError | None:
    if isinstance(exc, InspectError) and exc.kind == "invocation" and not rollback:
        return exc
    suffix = "no evidence was published"
    if rollback:
        suffix += f"; rollback incomplete: {rollback}"
    if handle is not None:
        # A pending or lost owner always has a bound process; its pid aids manual recovery.
        pid = handle._process.pid
        return InspectError(
            "cleanup_pending",
            f"radare2 child {pid} cleanup {handle.state}: {handle.detail}; "
            f"it may still need manual reaping; {suffix}",
        )
    if isinstance(exc, InspectError):
        kind, message = exc.kind, str(exc)
    elif isinstance(exc, KeyboardInterrupt):
        kind, message = "interrupted", "interrupted"
    elif isinstance(exc, TransportError):
        kind = "unavailable" if exc.code == "unavailable" else "failed"
        message = str(exc) if exc.code == "unavailable" else f"radare2 {exc.code}: {exc}"
    elif isinstance(exc, OSError):
        kind, message = "failed", exc.strerror or str(exc)
    else:
        return None
    return InspectError(kind, f"{message}; {suffix}")


def _inspect(target: str, out: str, interrupts: _Interrupts) -> str:
    session = workspace = tmp = final = None
    try:
        resolved, requested = _check_paths(target, out)
        before, digest, header = _observe(resolved)
        with _sigint_blocked():
            try:
                os.mkdir(requested, 0o700)
            except OSError as exc:
                raise InspectError("invocation", f"cannot create --out: {exc.strerror}") from exc
            workspace = requested
            os.chmod(workspace, 0o700)
        session = R2Session(resolved)
        with session:
            ij = session.metadata()
            remote = session.read_bytes(0, len(header)) if header else b""
        _verify(resolved, before, ij, header, remote)
        payload = _render(resolved, before.st_size, digest, header, ij)
        with _sigint_blocked():
            fd, tmp = tempfile.mkstemp(dir=workspace, prefix=".evidence.", suffix=".tmp")
            try:
                os.fchmod(fd, 0o600)
                _write_all(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)
            if signal.SIGINT in signal.sigpending():
                interrupts.ignoring = True  # Consume the pending signal on unblock.
                raise KeyboardInterrupt
            os.link(tmp, workspace / EVIDENCE_NAME)  # Never replaces an existing name.
            final = workspace / EVIDENCE_NAME
            os.unlink(tmp)
            tmp = None
            _fsync_directory(workspace)
            interrupts.ignoring = True  # Committed: a later SIGINT cannot undo success.
        return str(final)
    except BaseException as exc:
        interrupts.ignoring = True
        with _sigint_blocked():
            handle = getattr(exc, "cleanup", None)
            if handle is None and session is not None:
                handle = session.cleanup
            if handle is not None and handle.retry() == "complete":
                handle = None
            rollback = _rollback(final, tmp, workspace)
        error = _failure(exc, handle, rollback)
        if error is None or error is exc:
            raise
        raise error from exc


def run(target: str, out: str) -> str:
    """Inspect TARGET and publish OUT/evidence.json; return its path or raise InspectError.

    Must run on the main thread: SIGINT handling is installed for the run and
    restored afterwards. One bounded cleanup retry is made; a still-pending
    child is reported as cleanup_pending because process exit abandons it.
    """
    interrupts = _Interrupts()
    try:
        return _inspect(target, out, interrupts)
    finally:
        interrupts.restore()
