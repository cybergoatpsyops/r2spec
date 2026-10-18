"""Owner-controlled canonical obligation ledger with a generated, non-authoritative view.

A ledger directory holds ledger.json, the only authority, and LEDGER.md, which
is regenerated from it after every commit and never read back. Each mutation
locks the directory, checks the caller's expected revision, applies one
permitted change and atomically replaces ledger.json, so a reader sees either
the previous or the next complete revision. A retry after an unclear failure is
refused as stale rather than applied twice.

Covered means tests exist, not that they pass. Results are reported by the
owner; the ledger asserts only hashes it computed itself. Closure needs the
latest result at the current inputs to pass, unchanged evidence and an explicit
owner disposition. A declared input change reopens the named obligations, or
all of them, and an "unknown" change reopens them to pending. Passing tests
establish neither complete coverage nor semantic equivalence.

The ledger never imports the RE stack. Writers must run on the main thread
because SIGINT is held off around the commit.
"""
import contextlib
import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import signal
import stat

SCHEMA = "r2spec.ledger/1"
INSPECT_SCHEMA = "r2spec.re.inspect/1"
LEDGER_NAME = "ledger.json"
VIEW_NAME = "LEDGER.md"
STATES = ("pending", "specified", "covered", "closed")
INPUTS = ("target", "configuration", "specification", "tests")
EVIDENCE_KINDS = ("re-inspect", "test-result", "other")
OUTCOMES = ("pass", "fail", "error")
# The state a declared change lowers affected obligations to; "unknown" is the most conservative.
REOPEN_TO = {
    "target": "covered", "configuration": "covered", "tests": "specified",
    "specification": "pending", "unknown": "pending",
}
# Fields that only hold at or above a state; lowering below it clears them into history.
_HELD_FROM = (("closure", "closed"), ("coverage", "covered"), ("results", "covered"), ("specification", "specified"))
_TOP_FIELDS = {"history", "inputs", "location", "obligations", "revision", "schema"}
_RECORD_FIELDS = {"closure", "coverage", "evidence", "requirement", "results", "specification", "state", "unknowns"}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TEMP_PREFIXES = (".ledger.", ".view.")
_NOTE = (
    "Covered means tests exist, not that they pass. Passing tests establish neither "
    "complete coverage nor semantic equivalence."
)


class LedgerError(Exception):
    """A refused or failed ledger operation; kind selects the CLI exit status.

    Kinds: invocation, refused, failed, interrupted. committed is true only when
    the operation committed a revision before failing, and the message names it.
    """

    def __init__(self, kind: str, message: str, committed: bool = False):
        super().__init__(message)
        self.kind = kind
        self.committed = committed


def _invalid(message: str) -> LedgerError:
    return LedgerError("invocation", message)


def _refused(message: str) -> LedgerError:
    return LedgerError("refused", message)


def _reason(exc: OSError) -> str:
    return exc.strerror or str(exc)


def _check_id(value, what: str) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise _invalid(f"{what} must match {_ID.pattern}")


def _check_text(value, what: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise _invalid(f"{what} must be non-empty text")


@contextlib.contextmanager
def _sigint_blocked():
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _take_sigint() -> bool:
    """Consume a SIGINT held off by _sigint_blocked so it cannot fire once unblocked."""
    return signal.sigtimedwait({signal.SIGINT}, 0) is not None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_fd(fd: int) -> bytes:
    chunks = []
    while chunk := os.read(fd, 1 << 20):
        chunks.append(chunk)
    return b"".join(chunks)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _read_file(path, kind: str, what: str) -> tuple[str, bytes]:
    """Read an owner-supplied regular file at its resolved path without following a final symlink."""
    real = os.path.realpath(os.fspath(path))
    try:
        fd = os.open(real, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise LedgerError(kind, f"{what} is not a regular file: {real}")
            return real, _read_fd(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise LedgerError(kind, f"{what} cannot be read: {real}: {_reason(exc)}") from exc


def _open_directory(directory) -> int:
    """Open the ledger directory itself, refusing a symlink or a directory others can reach."""
    if os.path.islink(directory):
        raise _refused("the ledger directory must not be a symlink")
    try:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
            raise _invalid("the ledger directory is not an existing directory") from exc
        raise LedgerError("failed", f"cannot open the ledger directory: {_reason(exc)}") from exc
    info = os.fstat(fd)
    if info.st_uid != os.geteuid() or info.st_mode & 0o077:
        os.close(fd)
        raise _refused("the ledger directory must be owned by you and private (mode 0700)")
    return fd


def _check_location(directory, roots) -> None:
    real = os.path.realpath(directory)
    for root in roots:
        if os.path.commonpath([real, root]) == root:
            raise _refused(f"the ledger directory is inside forbidden root {root}")


def _is_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_sha(value) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _is_path(value) -> bool:
    return isinstance(value, str) and os.path.isabs(value)


def _shape(value, keys: set) -> bool:
    return isinstance(value, dict) and set(value) == keys


def _inputs_ok(value) -> bool:
    return _shape(value, set(INPUTS)) and all(_is_text(item) for item in value.values())


def _evidence_ok(entry) -> bool:
    if not _shape(entry, {"confidence", "kind", "path", "schema", "sha256", "target_sha256"}):
        return False
    if entry["kind"] == "re-inspect":
        bound = entry["schema"] == INSPECT_SCHEMA and _is_sha(entry["target_sha256"])
    else:
        bound = entry["kind"] in EVIDENCE_KINDS and entry["schema"] is None and entry["target_sha256"] is None
    return bound and entry["confidence"] == "observed" and _is_path(entry["path"]) and _is_sha(entry["sha256"])


def _result_ok(result, evidence: dict) -> bool:
    return (
        _shape(result, {"confidence", "evidence", "inputs", "outcome", "recorded_at"})
        and result["confidence"] == "reported" and isinstance(result["evidence"], str)
        and result["evidence"] in evidence and _inputs_ok(result["inputs"])
        and result["outcome"] in OUTCOMES and type(result["recorded_at"]) is int
    )


def _closure_ok(closure, results: list) -> bool:
    return (
        _shape(closure, {"closed_at", "disposition", "inputs", "limitations", "rationale", "result"})
        and closure["disposition"] == "accepted" and type(closure["closed_at"]) is int
        and _inputs_ok(closure["inputs"]) and _is_text(closure["rationale"])
        and isinstance(closure["limitations"], list) and all(_is_text(text) for text in closure["limitations"])
        and type(closure["result"]) is int and 0 <= closure["result"] < len(results)
    )


def _record_problem(record: dict) -> str | None:
    """Name the first problem in one obligation's fields or their fit with its state."""
    level = STATES.index(record["state"])
    evidence, coverage, results = record["evidence"], record["coverage"], record["results"]
    if not _is_text(record["requirement"]):
        return "requirement is not text"
    if not (record["specification"] is None if level < 1 else _is_text(record["specification"])):
        return "specification does not fit its state"
    if not isinstance(record["unknowns"], list) or not all(
        _shape(item, {"field", "reason"}) and _is_text(item["field"]) and _is_text(item["reason"])
        for item in record["unknowns"]
    ):
        return "unknowns are malformed"
    if not isinstance(evidence, dict) or not all(
        _ID.fullmatch(ref) and _evidence_ok(entry) for ref, entry in evidence.items()
    ):
        return "evidence is malformed"
    if level < 2:
        if coverage is not None or results != []:
            return "coverage or results do not fit its state"
    elif not (
        _shape(coverage, {"tests", "tests_input"}) and _is_text(coverage["tests_input"])
        and isinstance(coverage["tests"], list) and coverage["tests"] and all(
            _shape(test, {"path", "sha256"}) and _is_path(test["path"]) and _is_sha(test["sha256"])
            for test in coverage["tests"]
        )
    ):
        return "coverage is malformed"
    if not isinstance(results, list) or not all(_result_ok(result, evidence) for result in results):
        return "results are malformed"
    if not (record["closure"] is None if level < 3 else _closure_ok(record["closure"], results)):
        return "closure does not fit its state"
    return None


def _problem(document) -> str | None:
    """Name the first structural problem in a loaded ledger, or return None."""
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        return f"schema is not {SCHEMA}"
    if set(document) != _TOP_FIELDS:
        return "unexpected top-level fields"
    revision, inputs, location = document["revision"], document["inputs"], document["location"]
    if type(revision) is not int or revision < 0:
        return "revision is not a non-negative integer"
    if not isinstance(inputs, dict) or set(inputs) != set(INPUTS) or not all(
        value is None or (
            isinstance(value, dict) and set(value) == {"id", "since"}
            and _is_text(value["id"]) and type(value["since"]) is int
        )
        for value in inputs.values()
    ):
        return "inputs are malformed"
    if not isinstance(location, dict) or set(location) != {"forbidden_roots"} or not isinstance(
        location["forbidden_roots"], list,
    ) or not all(isinstance(root, str) and os.path.isabs(root) for root in location["forbidden_roots"]):
        return "location is malformed"
    if not isinstance(document["obligations"], dict):
        return "obligations are malformed"
    for key, record in document["obligations"].items():
        if not _ID.fullmatch(key) or not isinstance(record, dict) or set(record) != _RECORD_FIELDS or (
            record["state"] not in STATES
        ):
            return f"obligation {key!r} is malformed"
        problem = _record_problem(record)
        if problem:
            return f"obligation {key!r} {problem}"
    history = document["history"]
    if not isinstance(history, list) or not history or not isinstance(history[-1], dict) or (
        history[-1].get("revision") != revision
    ):
        return "history does not end at the current revision"
    return None


def _read_document(fd: int) -> tuple[dict, bytes]:
    try:
        file_fd = os.open(LEDGER_NAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=fd)
        try:
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                raise LedgerError("failed", f"corrupt ledger: {LEDGER_NAME} is not a regular file")
            data = _read_fd(file_fd)
        finally:
            os.close(file_fd)
    except OSError as exc:
        raise LedgerError("failed", f"cannot read {LEDGER_NAME}: {_reason(exc)}") from exc
    try:
        document = json.loads(data)
    except ValueError:  # Includes undecodable bytes.
        raise LedgerError("failed", f"corrupt ledger: {LEDGER_NAME} is not valid JSON") from None
    problem = _problem(document)
    if problem:
        raise LedgerError("failed", f"corrupt ledger: {problem}")
    return document, data


def _lock(fd: int) -> None:
    """Take the single-writer lock, then remove temp files a killed writer left behind."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise _refused("ledger busy: another writer holds the lock") from exc
    for name in os.listdir(fd):
        if name.startswith(_TEMP_PREFIXES) and name.endswith(".tmp") and stat.S_ISREG(
            os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode,
        ):
            os.unlink(name, dir_fd=fd)


@contextlib.contextmanager
def _locked(directory):
    """Yield the locked directory fd, its validated document and the document's bytes."""
    fd = _open_directory(directory)
    try:
        _lock(fd)
        document, data = _read_document(fd)
        _check_location(directory, document["location"]["forbidden_roots"])
        yield fd, document, data
    except OSError as exc:
        raise LedgerError("failed", f"{_reason(exc)}; ledger unchanged") from exc
    finally:
        os.close(fd)


def _write_temp(fd: int, prefix: str, data: bytes) -> str:
    name = f"{prefix}{secrets.token_hex(8)}.tmp"
    file_fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=fd)
    try:
        try:
            os.fchmod(file_fd, 0o600)
            _write_all(file_fd, data)
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(name, dir_fd=fd)
        raise
    return name


def _publish(fd: int, data: bytes, revision: int, unchanged: str | None) -> None:
    """Replace ledger.json with data, then regenerate LEDGER.md, with SIGINT held off.

    With unchanged=None only the view of already committed data is rewritten.
    Until os.replace returns nothing is visible and errors end with unchanged;
    after it the revision stands and errors name it.
    """
    view = _view(data)
    new = unchanged is not None
    committed = not new
    temp = None
    with _sigint_blocked():
        try:
            if new:
                temp = _write_temp(fd, ".ledger.", data)
                if _take_sigint():
                    raise LedgerError("interrupted", f"interrupted; {unchanged}")
                os.replace(temp, LEDGER_NAME, src_dir_fd=fd, dst_dir_fd=fd)
                temp, committed = None, True
                try:
                    os.fsync(fd)
                except OSError as exc:
                    raise LedgerError(
                        "failed", f"{_reason(exc)}; committed revision {revision}; durability unconfirmed", True,
                    ) from exc
            temp = _write_temp(fd, ".view.", view)
            os.replace(temp, VIEW_NAME, src_dir_fd=fd, dst_dir_fd=fd)
            temp = None
        except OSError as exc:
            if not committed:
                raise LedgerError("failed", f"{_reason(exc)}; {unchanged}") from exc
            raise LedgerError(
                "failed", f"{_reason(exc)}; committed revision {revision}; view not regenerated", new,
            ) from exc
        finally:
            if temp is not None:
                with contextlib.suppress(OSError):
                    os.unlink(temp, dir_fd=fd)
        if _take_sigint():
            raise LedgerError("interrupted", f"interrupted; committed revision {revision}", new)


def _dump(document: dict) -> bytes:
    return (json.dumps(document, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def _transaction(directory, expect_revision, apply) -> int:
    """Run apply(document, revision) under the lock and commit exactly one new revision."""
    if type(expect_revision) is not int or expect_revision < 0:
        raise _invalid("expect_revision must be a non-negative integer")
    with _locked(directory) as (fd, document, _):
        if document["revision"] != expect_revision:
            raise _refused(f"stale ledger revision: expected {expect_revision}, found {document['revision']}")
        revision = expect_revision + 1
        op, transitions, detail = apply(document, revision)
        document["revision"] = revision
        document["history"].append({"detail": detail, "op": op, "revision": revision, "transitions": transitions})
        _publish(fd, _dump(document), revision, f"ledger unchanged at revision {expect_revision}")
    return revision


def _current(document: dict) -> dict:
    return {name: value and value["id"] for name, value in document["inputs"].items()}


def _record(document: dict, obligation_id: str, op: str, *states: str) -> dict:
    record = document["obligations"].get(obligation_id)
    if record is None:
        raise _refused(f"unknown obligation {obligation_id}")
    if states and record["state"] not in states:
        raise _refused(f"{op} is not permitted for {obligation_id} in state {record['state']}")
    return record


def _move(obligation_id: str, old: str | None, new: str) -> dict:
    return {"from": old, "id": obligation_id, "to": new}


def _lower(record: dict, to: str) -> tuple[str, dict] | None:
    """Move a record down to state to; return its old state and the fields cleared, or None."""
    old = record["state"]
    if STATES.index(old) <= STATES.index(to):
        return None
    cleared = {}
    for field, held_from in _HELD_FROM:
        if STATES.index(to) < STATES.index(held_from) and record[field]:
            cleared[field] = record[field]
            record[field] = [] if field == "results" else None
    record["state"] = to
    return old, cleared


def init(directory, forbidden_roots=()) -> int:
    """Create a new private ledger directory at revision 0 and return 0.

    forbidden_roots are directories, such as an Implementer workspace, that the
    ledger must never be inside; every later operation checks them again.
    """
    if isinstance(forbidden_roots, (str, bytes, os.PathLike)):
        raise _invalid("forbidden_roots must be a list of paths")
    requested = os.path.abspath(directory)
    parent, name = os.path.split(requested)
    if not name or not os.path.isdir(parent):
        raise _invalid("the parent of the ledger directory must be an existing directory")
    if os.path.lexists(requested):
        raise _invalid("the ledger directory already exists; name a new directory")
    roots = sorted({os.path.realpath(root) for root in forbidden_roots})
    _check_location(requested, roots)
    data = _dump({
        "history": [{"detail": {}, "op": "init", "revision": 0, "transitions": []}],
        "inputs": dict.fromkeys(INPUTS), "location": {"forbidden_roots": roots},
        "obligations": {}, "revision": 0, "schema": SCHEMA,
    })
    try:
        with _sigint_blocked():  # A pending SIGINT is reported by _publish or fires after success.
            os.mkdir(requested, 0o700)
            try:
                os.chmod(requested, 0o700)
                fd = _open_directory(requested)
                try:
                    _lock(fd)
                    _publish(fd, data, 0, "no ledger was created")
                finally:
                    os.close(fd)
            except BaseException as exc:
                if not getattr(exc, "committed", False):
                    try:
                        os.rmdir(requested)
                    except OSError as cleanup:
                        detail = _reason(exc) if isinstance(exc, OSError) else str(exc)
                        raise LedgerError(
                            "failed", f"{detail}; could not remove {requested}: {_reason(cleanup)}",
                        ) from exc
                raise
    except OSError as exc:
        raise LedgerError("failed", f"{_reason(exc)}; no ledger was created") from exc
    return 0


def load(directory) -> tuple[dict, int, str]:
    """Return the committed document, its revision and the sha256 of its bytes.

    Reading takes no lock, because os.replace always leaves a whole revision.
    LEDGER.md is never read.
    """
    fd = _open_directory(directory)
    try:
        document, data = _read_document(fd)
    finally:
        os.close(fd)
    _check_location(directory, document["location"]["forbidden_roots"])
    return document, document["revision"], _sha256(data)


def render(directory) -> int:
    """Regenerate LEDGER.md from the committed ledger and return its revision."""
    with _locked(directory) as (fd, document, data):
        _publish(fd, data, document["revision"], None)
    return document["revision"]


def add(directory, obligation_id, requirement, *, expect_revision) -> int:
    """Add a pending obligation with its semantic requirement."""
    _check_id(obligation_id, "obligation id")
    _check_text(requirement, "requirement")

    def apply(document, revision):
        if obligation_id in document["obligations"]:
            raise _refused(f"obligation {obligation_id} already exists")
        document["obligations"][obligation_id] = {
            "closure": None, "coverage": None, "evidence": {}, "requirement": requirement,
            "results": [], "specification": None, "state": "pending", "unknowns": [],
        }
        return "add", [_move(obligation_id, None, "pending")], {}
    return _transaction(directory, expect_revision, apply)


def specify(directory, obligation_id, *, expect_revision) -> int:
    """Mark a pending obligation specified under the declared specification input."""
    _check_id(obligation_id, "obligation id")

    def apply(document, revision):
        record = _record(document, obligation_id, "specify", "pending")
        specification = document["inputs"]["specification"]
        if specification is None:
            raise _refused("declare a specification input before specifying obligations")
        record["state"], record["specification"] = "specified", specification["id"]
        return "specify", [_move(obligation_id, "pending", "specified")], {}
    return _transaction(directory, expect_revision, apply)


def attach_evidence(directory, obligation_id, ref, path, *, kind, expect_revision) -> int:
    """Reference a private evidence file by path and sha256; its content is never copied.

    re-inspect evidence must be a C1 r2spec.re.inspect/1 document for the declared target.
    """
    _check_id(obligation_id, "obligation id")
    _check_id(ref, "evidence ref")
    if kind not in EVIDENCE_KINDS:
        raise _invalid(f"evidence kind must be one of {', '.join(EVIDENCE_KINDS)}")
    real, data = _read_file(path, "invocation", "evidence file")
    entry = {
        "confidence": "observed", "kind": kind, "path": real, "schema": None,
        "sha256": _sha256(data), "target_sha256": None,
    }
    if kind == "re-inspect":
        try:
            evidence = json.loads(data)
            schema, target = evidence["schema"], evidence["target"]["sha256"]
        except (ValueError, TypeError, KeyError):
            schema = target = None
        if schema != INSPECT_SCHEMA or not isinstance(target, str) or not _SHA256.fullmatch(target):
            raise _invalid(f"re-inspect evidence must be an {INSPECT_SCHEMA} document")
        entry["schema"], entry["target_sha256"] = schema, target

    def apply(document, revision):
        record = _record(document, obligation_id, "attach_evidence")
        if ref in record["evidence"]:
            raise _refused(f"evidence {ref} is already attached to {obligation_id}")
        if kind == "re-inspect" and entry["target_sha256"] != _current(document)["target"]:
            raise _refused(f"evidence {ref} describes target {entry['target_sha256']}, not the declared target")
        record["evidence"][ref] = entry
        return "attach_evidence", [], {"id": obligation_id, "ref": ref}
    return _transaction(directory, expect_revision, apply)


def note_unknown(directory, obligation_id, field, reason, *, expect_revision) -> int:
    """Record an explicit unknown about an obligation."""
    _check_id(obligation_id, "obligation id")
    _check_text(field, "unknown field")
    _check_text(reason, "unknown reason")

    def apply(document, revision):
        record = _record(document, obligation_id, "note_unknown")
        if any(item["field"] == field for item in record["unknowns"]):
            raise _refused(f"an unknown for {json.dumps(field)} is already recorded on {obligation_id}")
        record["unknowns"] = sorted(
            [*record["unknowns"], {"field": field, "reason": reason}], key=lambda item: item["field"],
        )
        return "note_unknown", [], {"field": field, "id": obligation_id}
    return _transaction(directory, expect_revision, apply)


def cover(directory, obligation_id, tests, *, expect_revision) -> int:
    """Record that tests exist for a specified obligation; this says nothing about passing."""
    _check_id(obligation_id, "obligation id")
    if isinstance(tests, (str, bytes, os.PathLike)):
        raise _invalid("tests must be a list of test file paths")
    files = []
    for path in tests:
        real, data = _read_file(path, "invocation", "test file")
        files.append({"path": real, "sha256": _sha256(data)})
    if not files or len({item["path"] for item in files}) != len(files):
        raise _invalid("tests must name one or more distinct test files")

    def apply(document, revision):
        record = _record(document, obligation_id, "cover", "specified")
        tests_input = document["inputs"]["tests"]
        if tests_input is None:
            raise _refused("declare a tests input before covering obligations")
        record["state"], record["coverage"] = "covered", {"tests": files, "tests_input": tests_input["id"]}
        return "cover", [_move(obligation_id, "specified", "covered")], {}
    return _transaction(directory, expect_revision, apply)


def record_result(directory, obligation_id, *, outcome, evidence, inputs, expect_revision) -> int:
    """Record a reported test outcome bound to the inputs it ran against.

    inputs must name the current target, configuration, specification and tests
    ids, so a stale receipt is refused.
    """
    _check_id(obligation_id, "obligation id")
    _check_id(evidence, "evidence ref")
    if outcome not in OUTCOMES:
        raise _invalid(f"outcome must be one of {', '.join(OUTCOMES)}")
    if not isinstance(inputs, dict) or set(inputs) != set(INPUTS) or not all(
        isinstance(value, str) and value for value in inputs.values()
    ):
        raise _invalid(f"inputs must name all of {', '.join(INPUTS)}")

    def apply(document, revision):
        record = _record(document, obligation_id, "record_result", "covered")
        if evidence not in record["evidence"]:
            raise _refused(f"evidence {evidence} is not attached to {obligation_id}")
        if inputs != _current(document):
            raise _refused("result inputs do not match the current inputs; the result is stale")
        record["results"].append({
            "confidence": "reported", "evidence": evidence, "inputs": dict(inputs),
            "outcome": outcome, "recorded_at": revision,
        })
        return "record_result", [], {"id": obligation_id, "outcome": outcome}
    return _transaction(directory, expect_revision, apply)


def close(directory, obligation_id, *, expect_revision, disposition, rationale, limitations) -> int:
    """Close a covered obligation on the owner's explicit acceptance.

    The latest result at the current inputs must pass, its evidence and the
    covered test files must still hash as recorded, inspect evidence must
    describe the current target, and the obligation must be specified and
    covered under the current specification and tests inputs. limitations has
    no default; pass [] only when there are none.
    """
    _check_id(obligation_id, "obligation id")
    if disposition != "accepted":
        raise _invalid('closure needs the explicit disposition "accepted"')
    _check_text(rationale, "rationale")
    if not isinstance(limitations, (list, tuple)):
        raise _invalid("limitations must be a list; pass [] only when there are none")
    for text in limitations:
        _check_text(text, "each limitation")

    def apply(document, revision):
        record = _record(document, obligation_id, "close", "covered")
        current = _current(document)
        matching = [index for index, result in enumerate(record["results"]) if result["inputs"] == current]
        if not matching:
            raise _refused(f"{obligation_id} has no result at the current inputs")
        index = matching[-1]
        result = record["results"][index]
        if result["outcome"] != "pass":
            raise _refused(
                f"the latest result for {obligation_id} at the current inputs is {result['outcome']}, not pass",
            )
        if record["coverage"]["tests_input"] != current["tests"]:
            raise _refused(f"coverage of {obligation_id} predates the current tests input")
        if record["specification"] != current["specification"]:
            raise _refused(f"{obligation_id} was specified under a previous specification input")
        for test in record["coverage"]["tests"]:
            _, data = _read_file(test["path"], "refused", "covered test file")
            if _sha256(data) != test["sha256"]:
                raise _refused(f"covered test file {test['path']} changed since it was covered")
        ref = result["evidence"]
        entry = record["evidence"][ref]
        _, data = _read_file(entry["path"], "refused", f"evidence {ref}")
        if _sha256(data) != entry["sha256"]:
            raise _refused(f"evidence {ref} changed since it was attached")
        if entry["kind"] == "re-inspect" and entry["target_sha256"] != current["target"]:
            raise _refused(f"evidence {ref} describes target {entry['target_sha256']}, not the declared target")
        record["state"], record["closure"] = "closed", {
            "closed_at": revision, "disposition": disposition, "inputs": current,
            "limitations": list(limitations), "rationale": rationale, "result": index,
        }
        return "close", [_move(obligation_id, "covered", "closed")], {}
    return _transaction(directory, expect_revision, apply)


def change(directory, dimension, to_id, *, reason, affects=None, expect_revision) -> int:
    """Declare a changed input and reopen affected obligations (all unless named).

    target and configuration reopen to covered, tests to specified, and
    specification or unknown to pending. "unknown" leaves the inputs as they are.
    """
    if dimension not in REOPEN_TO:
        raise _invalid(f"dimension must be one of {', '.join(REOPEN_TO)}")
    _check_text(to_id, "input id")
    _check_text(reason, "reason")
    if dimension == "target" and not _SHA256.fullmatch(to_id):
        raise _invalid("the target id must be its lowercase sha256")
    if affects is not None:
        if isinstance(affects, str):
            raise _invalid("affects must be a list of obligation ids")
        affects = list(affects)
        for obligation_id in affects:
            _check_id(obligation_id, "obligation id")
        if not affects or len(set(affects)) != len(affects):
            raise _invalid("affects must name one or more distinct obligations")

    def apply(document, revision):
        obligations = document["obligations"]
        named = sorted(obligations) if affects is None else sorted(affects)
        for obligation_id in named:
            _record(document, obligation_id, "change")
        previous = None
        if dimension in INPUTS:
            previous = _current(document)[dimension]
            if previous == to_id:
                raise _refused(f"{dimension} is already {to_id}")
            document["inputs"][dimension] = {"id": to_id, "since": revision}
        to = REOPEN_TO[dimension]
        transitions, cleared = [], {}
        for obligation_id in named:
            moved = _lower(obligations[obligation_id], to)
            if moved:
                transitions.append(_move(obligation_id, moved[0], to))
                if moved[1]:
                    cleared[obligation_id] = moved[1]
        detail = {
            "affects": "all" if affects is None else sorted(affects), "dimension": dimension,
            "from": previous, "reason": reason, "to": to_id,
        }
        if cleared:
            detail["cleared"] = cleared
        return "change", transitions, detail
    return _transaction(directory, expect_revision, apply)


def reopen(directory, obligation_id, *, to, reason, expect_revision) -> int:
    """Owner reopen of one obligation to a strictly lower state."""
    _check_id(obligation_id, "obligation id")
    if to not in STATES[:-1]:
        raise _invalid(f"reopen target must be one of {', '.join(STATES[:-1])}")
    _check_text(reason, "reason")

    def apply(document, revision):
        record = _record(document, obligation_id, "reopen")
        state = record["state"]
        moved = _lower(record, to)
        if moved is None:
            raise _refused(f"reopen only moves {obligation_id} to a state below {state}")
        detail = {"reason": reason}
        if moved[1]:
            detail["cleared"] = {obligation_id: moved[1]}
        return "reopen", [_move(obligation_id, state, to)], detail
    return _transaction(directory, expect_revision, apply)


def _quote(text: str) -> str:
    return json.dumps(text, ensure_ascii=True)


def _view(data: bytes) -> bytes:
    """Render the non-authoritative Markdown view as a pure function of the ledger bytes."""
    document = json.loads(data)
    current = _current(document)
    lines = [
        f"<!-- GENERATED by r2spec from {LEDGER_NAME} revision {document['revision']} "
        f"(sha256 {_sha256(data)}). NOT AUTHORITATIVE. Edits are ignored and overwritten. -->",
        "# Obligation ledger (generated view)", "", _NOTE, "", "## Inputs", "",
    ]
    for name in INPUTS:
        value = document["inputs"][name]
        declared = "undeclared" if value is None else f"{_quote(value['id'])} since revision {value['since']}"
        lines.append(f"- {name}: {declared}")
    lines += ["", "## Obligations"]
    if not document["obligations"]:
        lines += ["", "None recorded."]
    for obligation_id, record in sorted(document["obligations"].items()):
        specification = record["specification"]
        lines += [
            "", f"### {obligation_id}: {record['state']}", "",
            f"- requirement: {_quote(record['requirement'])}",
            f"- specification: {'none' if specification is None else _quote(specification)}",
            "- unknowns:" + ("" if record["unknowns"] else " none recorded"),
        ]
        lines += [f"  - {_quote(item['field'])}: {_quote(item['reason'])}" for item in record["unknowns"]]
        lines.append("- evidence:" + ("" if record["evidence"] else " none attached"))
        for ref, entry in sorted(record["evidence"].items()):
            target = "" if entry["target_sha256"] is None else f", target {entry['target_sha256']}"
            lines.append(
                f"  - {ref}: {entry['kind']}, sha256 {entry['sha256']} ({entry['confidence']}){target}, "
                f"{_quote(entry['path'])}",
            )
        coverage = record["coverage"]
        if coverage is None:
            lines.append("- coverage: none")
        else:
            lines.append(f"- coverage: tests exist for tests input {_quote(coverage['tests_input'])}; not a pass")
            lines += [f"  - {_quote(test['path'])} sha256 {test['sha256']}" for test in coverage["tests"]]
        lines.append("- results:" + ("" if record["results"] else " none recorded"))
        for index, result in enumerate(record["results"]):
            binding = "current inputs" if result["inputs"] == current else "stale inputs"
            lines.append(
                f"  - #{index} {result['outcome']} ({result['confidence']}) at revision "
                f"{result['recorded_at']}, evidence {result['evidence']}, {binding}",
            )
        closure = record["closure"]
        if closure is None:
            lines.append("- closure: none")
        else:
            lines += [
                f"- closure: {closure['disposition']} at revision {closure['closed_at']} on result "
                f"#{closure['result']}: {_quote(closure['rationale'])}",
                "  - limitations:" + ("" if closure["limitations"] else " none declared by the owner"),
            ]
            lines += [f"    - {_quote(text)}" for text in closure["limitations"]]
    return ("\n".join(lines) + "\n").encode("ascii")
