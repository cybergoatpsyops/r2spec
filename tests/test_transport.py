"""Real process-boundary tests.

Failure modes: missing executable; invalid paths/parameters; startup and request
stalls; oversized, missing, extra, invalid or wrong-shape frames; closed pipes;
process death; cancellation; source changes; leaked processes/descriptors.
Fake executables substitute only for the external r2 process.
"""
import hashlib
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from r2spec import _child, transport as transport_module
from r2spec.transport import R2Session, TransportError


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "bytes with spaces;literal.bin"
    path.write_bytes(bytes(range(32)))
    return path


def make_peer(tmp_path, body):
    executable = tmp_path / "peer"
    pid_file = tmp_path / "peer.pid"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os, sys, time, signal\n"
        "from pathlib import Path\n"
        f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        + textwrap.dedent(body)
    )
    executable.chmod(0o700)
    return executable, pid_file


def assert_reaped(pid_file):
    assert pid_file.exists(), "external peer never started"
    assert not Path("/proc", pid_file.read_text().strip()).exists()


def test_real_bytes_are_exact_and_original_unchanged(sample):
    before = hashlib.sha256(sample.read_bytes()).hexdigest()
    before_fds = len(os.listdir("/proc/self/fd"))
    with R2Session(sample) as session:
        assert session.read_bytes(4, 4) == bytes(range(4, 8))
        pid = session.pid
        assert session.read_bytes(0, 3) == b"\0\1\2"
        assert session.pid == pid
        assert session.metadata()["core"]["size"] == 32
        assert session.read_bytes(4, 4) == bytes(range(4, 8))
    assert not Path(f"/proc/{pid}").exists()
    assert len(os.listdir("/proc/self/fd")) == before_fds
    assert hashlib.sha256(sample.read_bytes()).hexdigest() == before


def test_native_elf_is_inspected_not_executed(tmp_path):
    assert platform.machine() == "aarch64"
    target = tmp_path / "native"
    subprocess.run(
        ["cc", "-g", "-O0", str(Path(__file__).parent / "fixtures/native.c"),
         "-o", str(target)], check=True, capture_output=True, timeout=20,
    )
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    with R2Session(target) as session:
        assert session.read_bytes(0, 4) == b"\x7fELF"
        info = session.metadata()
        assert info["bin"]["arch"] == "arm"
        assert info["bin"]["bits"] == 64
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("offset,count", [
    (-1, 1), (True, 1), (0, True), (0, 0), (0, -1),
    (0, 65537), (2**64, 1), (31, 2), ("0;quit", 1),
])
def test_invalid_read_does_not_touch_or_destroy_session(sample, offset, count):
    with R2Session(sample) as session:
        with pytest.raises(TransportError) as error:
            session.read_bytes(offset, count)
        assert error.value.code == "input"
        assert session.read_bytes(0, 1) == b"\0"


@pytest.mark.parametrize("kind", ["missing", "directory", "fifo", "newline"])
def test_non_regular_or_control_character_path_is_rejected(tmp_path, kind):
    path = tmp_path / ("bad\nname" if kind == "newline" else kind)
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "newline":
        path.write_bytes(b"x")
    with pytest.raises(TransportError) as error:
        R2Session(path)
    assert error.value.code == "input"


@pytest.mark.parametrize("options", [
    {"timeout": 0}, {"startup_timeout": float("inf")},
    {"timeout": float("nan")}, {"timeout": True},
    {"max_response_bytes": 0}, {"max_response_bytes": True},
])
def test_invalid_limits_are_rejected_before_spawn(sample, options):
    with pytest.raises(TransportError) as error:
        R2Session(sample, **options)
    assert error.value.code == "input"


def test_missing_executable(sample, tmp_path):
    with pytest.raises(TransportError) as error:
        R2Session(sample, executable=tmp_path / "absent")
    assert error.value.code == "unavailable"


@pytest.mark.parametrize("body,expected", [
    ("time.sleep(30)", "startup_timeout"),
    ("sys.stdout.buffer.write(b'not-a-handshake\\0'); sys.stdout.flush(); time.sleep(30)", "protocol"),
    ("sys.stdout.buffer.write(b'\\0extra'); sys.stdout.flush(); time.sleep(30)", "protocol"),
    ("sys.stdout.buffer.write(b'x'*4096); sys.stdout.flush(); time.sleep(30)", "overflow"),
    ("sys.exit(3)", "process"),
])
def test_startup_failures_are_bounded_and_reaped(sample, tmp_path, body, expected):
    peer, pid_file = make_peer(tmp_path, body)
    start = time.monotonic()
    with pytest.raises(TransportError) as error:
        R2Session(sample, executable=peer, startup_timeout=0.25, max_response_bytes=64)
    assert error.value.code == expected
    assert time.monotonic() - start < 3
    assert_reaped(pid_file)


@pytest.mark.parametrize("reply,expected", [
    (b"\0", "json"), (b"{\0", "json"), (b"\xff\0", "json"),
    (b"NaN\0", "json"), (b"{}\0", "schema"), (b"[true]\0", "schema"),
    (b"[-1]\0", "schema"), (b"[256]\0", "schema"),
    (b"[0,1]\0", "schema"), (b"[]\0extra", "protocol"),
    (b"x" * 4096, "overflow"),
])
def test_bad_responses_invalidate_session(sample, tmp_path, reply, expected):
    peer, pid_file = make_peer(tmp_path, f"""
        sys.stdout.buffer.write(b'\\0'); sys.stdout.flush()
        sys.stdin.buffer.readline()
        sys.stdout.buffer.write({reply!r}); sys.stdout.flush()
        time.sleep(30)
    """)
    with R2Session(sample, executable=peer, max_response_bytes=64) as session:
        with pytest.raises(TransportError) as error:
            session.read_bytes(0, 1)
        assert error.value.code == expected
        with pytest.raises(TransportError) as closed:
            session.read_bytes(0, 1)
        assert closed.value.code == "closed"
    assert_reaped(pid_file)


@pytest.mark.parametrize("reply,expected", [
    (b'{"core":{"file":"x","size":true}}\0', "schema"),
    (b'{"core":{"file":"x","size":32},"bin":[]}\0', "schema"),
    (b'{"core":{"file":"x","size":32},"core":{}}\0', "json"),
])
def test_metadata_has_strict_json_and_core_shape(sample, tmp_path, reply, expected):
    peer, pid_file = make_peer(tmp_path, f"""
        sys.stdout.buffer.write(b'\\0'); sys.stdout.flush()
        sys.stdin.buffer.readline()
        sys.stdout.buffer.write({reply!r}); sys.stdout.flush()
        time.sleep(30)
    """)
    with R2Session(sample, executable=peer) as session:
        with pytest.raises(TransportError) as error:
            session.metadata()
        assert error.value.code == expected
    assert_reaped(pid_file)


@pytest.mark.parametrize("action,expected", [
    ("time.sleep(30)", "timeout"),
    ("sys.stdout.buffer.write(b'[0]'); sys.stdout.flush(); time.sleep(30)", "timeout"),
    ("sys.exit(4)", "process"),
    ("os.close(1); time.sleep(30)", "process"),
])
def test_request_faults_terminate_and_reap(sample, tmp_path, action, expected):
    peer, pid_file = make_peer(tmp_path, f"""
        sys.stdout.buffer.write(b'\\0'); sys.stdout.flush()
        sys.stdin.buffer.readline()
        {action}
    """)
    with R2Session(sample, executable=peer, timeout=0.2) as session:
        with pytest.raises(TransportError) as error:
            session.read_bytes(0, 1)
        assert error.value.code == expected
    assert_reaped(pid_file)


def test_closed_stdin_is_a_process_error(sample, tmp_path):
    peer, pid_file = make_peer(tmp_path, """
        os.close(0)
        sys.stdout.buffer.write(b'\\0'); sys.stdout.flush()
        time.sleep(30)
    """)
    with R2Session(sample, executable=peer) as session:
        with pytest.raises(TransportError) as error:
            session.read_bytes(0, 1)
        assert error.value.code == "process"
    assert_reaped(pid_file)


def test_cancellation_while_waiting(sample, tmp_path):
    peer, pid_file = make_peer(tmp_path, """
        sys.stdout.buffer.write(b'\\0'); sys.stdout.flush()
        sys.stdin.buffer.readline()
        time.sleep(30)
    """)
    cancel = threading.Event()
    with R2Session(sample, executable=peer, timeout=5, cancel=cancel) as session:
        timer = threading.Timer(0.15, cancel.set)
        timer.start()
        try:
            with pytest.raises(TransportError) as error:
                session.read_bytes(0, 1)
            assert error.value.code == "cancelled"
        finally:
            timer.cancel()
            timer.join()
    assert_reaped(pid_file)


def test_startup_cancellation(sample, tmp_path):
    peer, pid_file = make_peer(tmp_path, "time.sleep(30)")
    cancel = threading.Event()
    timer = threading.Timer(0.15, cancel.set)
    timer.start()
    try:
        with pytest.raises(TransportError) as error:
            R2Session(sample, executable=peer, startup_timeout=5, cancel=cancel)
        assert error.value.code == "cancelled"
    finally:
        timer.cancel()
        timer.join()
    assert_reaped(pid_file)


def test_stopped_real_r2_is_killed_and_reaped(sample):
    with R2Session(sample, timeout=0.15) as session:
        assert session.read_bytes(0, 1) == b"\0"
        pid = session.pid
        os.kill(pid, signal.SIGSTOP)
        with pytest.raises(TransportError) as error:
            session.read_bytes(0, 1)
        assert error.value.code == "timeout"
    assert not Path(f"/proc/{pid}").exists()


def test_close_escalates_when_peer_ignores_term(sample, tmp_path):
    peer, pid_file = make_peer(tmp_path, """
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        sys.stdout.buffer.write(b'\\0'); sys.stdout.flush()
        time.sleep(30)
    """)
    start = time.monotonic()
    session = R2Session(sample, executable=peer)
    session.close()
    session.close()
    assert time.monotonic() - start < 3
    assert_reaped(pid_file)


def test_context_exit_on_caller_error_reaps_real_r2(sample):
    with pytest.raises(ValueError):
        with R2Session(sample) as session:
            pid = session.pid
            raise ValueError("caller failure")
    assert not Path(f"/proc/{pid}").exists()


def test_source_change_invalidates_session(sample):
    with R2Session(sample) as session:
        pid = session.pid
        sample.write_bytes(b"changed")
        with pytest.raises(TransportError) as error:
            session.read_bytes(0, 1)
        assert error.value.code == "source_changed"
    assert not Path(f"/proc/{pid}").exists()


def test_process_killed_before_request_is_not_reused(sample):
    with R2Session(sample) as session:
        pid = session.pid
        os.kill(pid, signal.SIGKILL)
        with pytest.raises(TransportError) as error:
            session.read_bytes(0, 1)
        assert error.value.code == "process"
    assert not Path(f"/proc/{pid}").exists()


# Lifecycle failure modes: close races with I/O; cleanup masks the primary;
# failed signaling discards ownership; premature reap allows stale group signals.
# Signal failures are injected at the OS boundary, not claimed kernel failures.
def wait_for_file(path):
    deadline = time.monotonic() + 3
    while not path.exists():
        assert time.monotonic() < deadline, f"peer did not signal {path.name}"
        time.sleep(0.005)


def force_reap(process, killpg):
    """Fixture recovery only; never signal after ownership has been lost."""
    try:
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return
    try:
        killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=3)
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


# Setup failure isolation: owner allocation must precede spawn; setup after
# binding must preserve BaseException identity and retain incomplete cleanup.
# A real failed exec returns no handle and must not be masked by empty teardown.
@pytest.fixture
def transport_spawns(monkeypatch):
    real_popen, real_killpg = subprocess.Popen, os.killpg
    spawned = []
    def record_spawn(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        spawned.append(process)
        return process
    monkeypatch.setattr(subprocess, "Popen", record_spawn)
    yield spawned
    for process in spawned:
        force_reap(process, real_killpg)
        for stream in (process.stdin, process.stdout):
            stream.close()


def test_transport_owner_allocation_failure_precedes_spawn(
    sample, transport_spawns, monkeypatch,
):
    primary = MemoryError("injected owner allocation failure, not real exhaustion")
    calls = []
    def failed_owner(*args, **kwargs):
        calls.append(True)
        raise primary
    monkeypatch.setattr(transport_module, "ChildCleanup", failed_owner)
    with pytest.raises(MemoryError) as caught:
        R2Session(sample)
    assert caught.value is primary
    assert getattr(primary, "cleanup", None) is None
    assert calls == [True]
    assert transport_spawns == [], "child spawned before cleanup owner existed"


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_transport_post_binding_setup_preserves_primary_and_owner(
    sample, transport_spawns, monkeypatch, pending, error_type,
):
    primary = error_type("injected nonblocking-pipe setup failure")
    def failed_setup(*args):
        raise primary
    def denied(*args):
        raise PermissionError("injected group-signal refusal")
    with monkeypatch.context() as patch:
        patch.setattr(transport_module.os, "set_blocking", failed_setup)
        if pending:
            patch.setattr(_child, "_signal_group", denied)
        with pytest.raises(error_type) as caught:
            R2Session(sample)
        assert caught.value is primary
        assert len(transport_spawns) == 1
        process = transport_spawns[0]
        assert process.stdin.closed and process.stdout.closed
        owner = getattr(primary, "cleanup", None)
        if pending:
            assert owner is not None and owner.state == "pending"
            assert process.returncode is None
            os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        else:
            assert owner is None
    if pending:
        assert owner.retry() == "complete"
        assert owner.retry() == "complete"
    assert process.returncode is not None
    with pytest.raises(ChildProcessError):
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)


def test_transport_exec_failure_preserves_unavailable_without_handle(sample, tmp_path):
    broken = tmp_path / "broken-r2"
    broken.write_text(f"#!{tmp_path / 'missing-interpreter'}\n")
    broken.chmod(0o700)
    before_children = Path(f"/proc/self/task/{os.getpid()}/children").read_text()
    before_fds = len(os.listdir("/proc/self/fd"))
    with pytest.raises(TransportError) as caught:
        R2Session(sample, executable=broken)
    assert caught.value.code == "unavailable"
    assert isinstance(caught.value.__cause__, FileNotFoundError)
    assert caught.value.cleanup is None
    assert Path(f"/proc/self/task/{os.getpid()}/children").read_text() == before_children
    assert len(os.listdir("/proc/self/fd")) == before_fds


@pytest.mark.parametrize("action", ["close", "busy"])
def test_concurrent_close_or_busy_preserves_typed_outcomes(sample, tmp_path, action):
    received, release = tmp_path / "received", tmp_path / "release"
    peer, _ = make_peer(tmp_path, f"""
        sys.stdout.buffer.write(b'\\0'); sys.stdout.flush()
        sys.stdin.buffer.readline()
        Path({str(received)!r}).touch()
        while not Path({str(release)!r}).exists(): time.sleep(0.005)
        sys.stdout.buffer.write(b'[0]\\0'); sys.stdout.flush()
        sys.stdin.buffer.readline()
        sys.stdout.buffer.write(b'[0]\\0'); sys.stdout.flush()
        sys.stdin.buffer.readline()
    """)
    session = R2Session(sample, executable=peer, timeout=5)
    outcomes = []
    def request():
        try:
            outcomes.append(session.read_bytes(0, 1))
        except BaseException as exc:
            outcomes.append(exc)
    first = threading.Thread(target=request)
    second = None
    first.start()
    try:
        wait_for_file(received)
        if action == "close":
            started = time.monotonic()
            session.close()
            assert time.monotonic() - started < 3
            first.join(3)
            assert not first.is_alive()
            assert len(outcomes) == 1
            assert isinstance(outcomes[0], TransportError)
            assert outcomes[0].code == "closed"
            with pytest.raises(TransportError) as error:
                session.metadata()
            assert error.value.code == "closed"
            session.close()
        else:
            second = threading.Thread(target=request)
            second.start()
            second.join(0.5)
            assert not second.is_alive(), "busy request blocked instead of rejecting"
            assert isinstance(outcomes[0], TransportError)
            assert outcomes[0].code == "busy"
            release.touch()
            first.join(3)
            assert not first.is_alive()
            assert outcomes[1] == b"\0"
            assert session.read_bytes(0, 1) == b"\0"
    finally:
        release.touch()
        session.close()
        first.join(3)
        if second is not None:
            second.join(3)


@pytest.mark.parametrize("where", ["constructor", "operation", "context", "bare_context", "close"])
def test_cleanup_failure_retains_owner_and_primary(sample, tmp_path, monkeypatch, where):
    body = (
        "sys.stdout.buffer.write(b'{\\0'); sys.stdout.flush(); sys.stdin.buffer.read()"
        if where == "constructor" else
        "sys.stdout.buffer.write(b'\\0'); sys.stdout.flush()\n"
        "sys.stdin.buffer.readline()\n"
        "sys.stdout.buffer.write(b'{\\0'); sys.stdout.flush(); sys.stdin.buffer.read()"
    )
    peer, _ = make_peer(tmp_path, body)
    spawned = []
    real_popen, real_killpg = subprocess.Popen, os.killpg
    def record_spawn(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        spawned.append(process)
        return process
    def deny_signal(*args):
        raise PermissionError("simulated group-signal failure")
    primary = ValueError("caller failure")
    session = None
    try:
        with monkeypatch.context() as patch:
            patch.setattr(subprocess, "Popen", record_spawn)
            if where == "constructor":
                patch.setattr(_child, "_signal_group", deny_signal)
                with pytest.raises(TransportError) as caught:
                    R2Session(sample, executable=peer)
                expected = "protocol"
            elif where == "bare_context":
                patch.setattr(_child, "_signal_group", deny_signal)
                with pytest.raises(ValueError) as caught:
                    with R2Session(sample, executable=peer):
                        raise primary
                assert caught.value is primary
                expected = None
            else:
                session = R2Session(sample, executable=peer)
                patch.setattr(_child, "_signal_group", deny_signal)
                if where == "context":
                    with pytest.raises(ValueError) as caught:
                        with session:
                            raise primary
                    assert caught.value is primary
                    expected = None
                elif where == "operation":
                    with pytest.raises(TransportError) as caught:
                        session.read_bytes(0, 1)
                    expected = "json"
                else:
                    with pytest.raises(TransportError) as caught:
                        session.close()
                    expected = "cleanup"
            if expected is not None:
                assert caught.value.code == expected
            owner = caught.value.cleanup
            assert owner is not None and owner.state == "pending"
            assert caught.value.__notes__
            if session is not None:
                assert session.cleanup is owner
            assert spawned[0].returncode is None, "failed cleanup reaped the identity pin"
        assert owner.retry() == "complete"
        assert not Path(f"/proc/{spawned[0].pid}").exists()
        assert owner.retry() == "complete"
        if session is not None:
            session.close()
            assert session.cleanup is None
    finally:
        for process in spawned:
            force_reap(process, real_killpg)


def test_signals_precede_reap(sample, monkeypatch):
    real_killpg = os.killpg
    signals = []
    def checked_signal(pid, sig):
        # Raises ECHILD before forwarding an unsafe call, including in the old code.
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        signals.append(sig)
        return real_killpg(pid, sig)
    with monkeypatch.context() as patch:
        patch.setattr(_child, "_signal_group", checked_signal)
        with R2Session(sample):
            pass
    assert signals[-1] == signal.SIGKILL


@pytest.mark.parametrize("symlink", [False, True])
def test_real_metadata_identity_characterization(sample, tmp_path, symlink):
    target = sample
    if symlink:
        target = tmp_path / "alias"
        target.symlink_to(sample)
    with R2Session(target) as session:
        core = session.metadata()["core"]
        assert core["file"] == str(sample.resolve())
        assert core["size"] == sample.stat().st_size


@pytest.mark.parametrize("field", ["file", "size"])
def test_wrong_metadata_identity_invalidates_session(sample, tmp_path, field):
    import json
    core = {"file": str(sample.resolve()), "size": sample.stat().st_size}
    core[field] = "other-file" if field == "file" else 99
    reply = json.dumps({"core": core}).encode() + b"\0"
    peer, pid_file = make_peer(tmp_path, f"""
        sys.stdout.buffer.write(b'\\0'); sys.stdout.flush()
        sys.stdin.buffer.readline()
        sys.stdout.buffer.write({reply!r}); sys.stdout.flush()
        sys.stdin.buffer.read()
    """)
    with R2Session(sample, executable=peer) as session:
        with pytest.raises(TransportError) as error:
            session.metadata()
        assert error.value.code == "schema"
        with pytest.raises(TransportError) as closed:
            session.read_bytes(0, 1)
        assert closed.value.code == "closed"
    assert_reaped(pid_file)


# Framing regressions: exact-byte limits, partial frames and delayed unsolicited
# output. The read spy below observes the real pipe boundary solely to release
# the peer after the first chunk has actually been read, not after a guessed sleep.
@pytest.mark.parametrize("length,expected", [(64, None), (65, "overflow")])
def test_exact_response_byte_boundary(sample, tmp_path, length, expected):
    reply = b"[0]" + b" " * (length - 3) + b"\0"
    peer, pid_file = make_peer(tmp_path, f"""
        sys.stdout.buffer.write(b'\\0'); sys.stdout.flush()
        sys.stdin.buffer.readline()
        sys.stdout.buffer.write({reply!r}); sys.stdout.flush()
        sys.stdin.buffer.read()
    """)
    with R2Session(sample, executable=peer, max_response_bytes=64) as session:
        if expected:
            with pytest.raises(TransportError) as error:
                session.read_bytes(0, 1)
            assert error.value.code == expected
        else:
            assert session.read_bytes(0, 1) == b"\0"
    assert_reaped(pid_file)


def test_large_fragmented_frame_waits_for_terminator(sample, tmp_path, monkeypatch):
    release = tmp_path / "first-chunk-read"
    peer, pid_file = make_peer(tmp_path, f"""
        sys.stdout.buffer.write(b'\\0'); sys.stdout.flush()
        sys.stdin.buffer.readline()
        sys.stdout.buffer.write(b'['); sys.stdout.flush()
        while not Path({str(release)!r}).exists(): time.sleep(0.005)
        sys.stdout.buffer.write(b' '*200000+b'0]\\0'); sys.stdout.flush()
        sys.stdin.buffer.read()
    """)
    with R2Session(sample, executable=peer, max_response_bytes=200003) as session:
        fd = session._process.stdout.fileno()
        real_read = os.read
        chunks = []
        def observed_read(target, count):
            chunk = real_read(target, count)
            if target == fd and chunk:
                chunks.append(chunk)
                release.touch()
            return chunk
        with monkeypatch.context() as patch:
            patch.setattr(os, "read", observed_read)
            assert session.read_bytes(0, 1) == b"\0"
        assert len(chunks) > 2
        assert chunks[0] == b"["
        assert sum(map(len, chunks)) == 200004
    assert_reaped(pid_file)


def test_delayed_unsolicited_output_cannot_answer_next_request(sample, tmp_path):
    # The extra frame is already queued before the next write. This does not
    # establish provenance for an unsolicited frame arriving after that write.
    release, written = tmp_path / "release-extra", tmp_path / "extra-written"
    peer, pid_file = make_peer(tmp_path, f"""
        sys.stdout.buffer.write(b'\\0'); sys.stdout.flush()
        sys.stdin.buffer.readline()
        sys.stdout.buffer.write(b'[0]\\0'); sys.stdout.flush()
        while not Path({str(release)!r}).exists(): time.sleep(0.005)
        sys.stdout.buffer.write(b'[1]\\0'); sys.stdout.flush()
        Path({str(written)!r}).touch()
        sys.stdin.buffer.read()
    """)
    with R2Session(sample, executable=peer) as session:
        assert session.read_bytes(0, 1) == b"\0"
        release.touch()
        wait_for_file(written)
        with pytest.raises(TransportError) as error:
            session.read_bytes(1, 1)
        assert error.value.code == "protocol"
        with pytest.raises(TransportError) as closed:
            session.read_bytes(0, 1)
        assert closed.value.code == "closed"
    assert_reaped(pid_file)
