"""Owned-child boundary checks with real subprocesses.

Isolation is needed for OS failures not safely inducible on demand. Failure modes:
reaping before final group signaling; treating failed signaling as success;
losing a pending owner; signaling after ECHILD; deadlines that discard identity;
replay after completion; failed wait/reap; primary errors replaced by diagnostics.
Only OS signaling/wait boundaries are injected. An unbound owner must also finish
without any process syscalls. No PID-reuse or D-state claim.
"""
import errno
import os
import signal
import subprocess
import sys
import time

import pytest

from r2spec import _child
from r2spec._child import ChildCleanup


@pytest.fixture
def process():
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", "import sys; sys.stdin.buffer.read()"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    killpg, waitpid = os.killpg, os.waitpid
    yield child
    try:
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        pass
    else:
        try:
            killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _, status = waitpid(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
    child.stdin.close()


def exited_without_reap(process):
    deadline = time.monotonic() + 3
    while True:
        info = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if info is not None:
            return info
        assert time.monotonic() < deadline
        time.sleep(0.005)


@pytest.mark.parametrize("first", ["finish", "retry"])
def test_unbound_owner_completes_without_process_syscalls(monkeypatch, first):
    def forbidden(*args):
        pytest.fail("unbound owner used a process syscall")
    monkeypatch.setattr(_child, "_signal_group", forbidden)
    monkeypatch.setattr(os, "waitid", forbidden)
    monkeypatch.setattr(os, "waitpid", forbidden)
    owner = ChildCleanup()
    assert getattr(owner, first)() == "complete"
    assert owner.finish() == owner.retry() == "complete"
    assert owner.detail == ""


def test_observation_pins_exit_and_finish_reaps_once(process, monkeypatch):
    owner = ChildCleanup(process)
    process.stdin.close()
    exited_without_reap(process)
    assert owner.observe() == "exited"
    assert owner.observe() == "exited"
    assert process.returncode is None
    real_signal = _child._signal_group
    calls = []
    def signal_pinned(pid, sig):
        assert os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
        calls.append(sig)
        return real_signal(pid, sig)
    monkeypatch.setattr(_child, "_signal_group", signal_pinned)
    assert owner.finish() == "complete"
    assert calls[-1] == signal.SIGKILL
    assert owner.returncode == 0
    with pytest.raises(ChildProcessError):
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    before = list(calls)
    assert owner.finish() == owner.retry() == "complete"
    assert calls == before


def test_final_signal_failure_keeps_zombie_for_explicit_retry(process, monkeypatch):
    owner = ChildCleanup(process)
    process.stdin.close()
    exited_without_reap(process)
    def denied(pid, sig):
        raise PermissionError(errno.EPERM, "simulated signal refusal")
    with monkeypatch.context() as patch:
        patch.setattr(_child, "_signal_group", denied)
        assert owner.finish() == "pending"
        assert owner.detail
        assert owner.returncode is None
        assert exited_without_reap(process).si_pid == process.pid
    assert owner.retry() == "complete"
    assert owner.returncode == 0


def test_term_failure_does_not_poison_successful_kill(process, monkeypatch):
    owner = ChildCleanup(process)
    real_signal = _child._signal_group
    def term_denied(pid, sig):
        if sig == signal.SIGTERM:
            raise PermissionError(errno.EPERM, "simulated TERM refusal")
        return real_signal(pid, sig)
    monkeypatch.setattr(_child, "_signal_group", term_denied)
    assert owner.finish() == "complete"
    assert owner.returncode == -signal.SIGKILL


def test_missing_group_can_complete_only_with_observed_and_reaped_leader(process, monkeypatch):
    owner = ChildCleanup(process)
    process.stdin.close()
    exited_without_reap(process)
    def missing(pid, sig):
        raise ProcessLookupError(errno.ESRCH, "simulated missing group")
    monkeypatch.setattr(_child, "_signal_group", missing)
    assert owner.finish() == "complete"
    assert owner.returncode == 0


def test_external_reap_is_lost_and_never_signaled_again(process, monkeypatch):
    owner = ChildCleanup(process)
    process.stdin.close()
    exited_without_reap(process)
    os.waitpid(process.pid, 0)  # Deliberate violation of exclusive-reaper precondition.
    def forbidden(*args):
        pytest.fail("numeric signal after lost ownership")
    monkeypatch.setattr(_child, "_signal_group", forbidden)
    assert owner.observe() == "lost"
    assert owner.finish() == owner.retry() == "lost"
    assert owner.detail


def test_unobserved_exit_is_pending_not_success(process, monkeypatch):
    owner = ChildCleanup(process)
    real_waitid = os.waitid
    def not_yet(*args):
        return None
    with monkeypatch.context() as patch:
        patch.setattr(os, "waitid", not_yet)
        started = time.monotonic()
        assert owner.finish() == "pending"
        assert time.monotonic() - started < 2
        assert owner.returncode is None
    assert real_waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    assert owner.retry() == "complete"


@pytest.mark.parametrize("boundary", ["waitid", "waitpid"])
def test_wait_failure_retains_owner(process, monkeypatch, boundary):
    owner = ChildCleanup(process)
    process.stdin.close()
    exited_without_reap(process)
    def failed(*args):
        raise OSError(errno.EIO, "simulated wait failure")
    with monkeypatch.context() as patch:
        patch.setattr(os, boundary, failed)
        assert owner.finish() == "pending"
        assert owner.returncode is None
        assert owner.detail
    assert owner.retry() == "complete"


def test_reap_loss_after_final_signal_cannot_become_success(process, monkeypatch):
    owner = ChildCleanup(process)
    process.stdin.close()
    exited_without_reap(process)
    real_waitpid = os.waitpid
    def stolen(pid, flags):
        real_waitpid(pid, flags)
        raise ChildProcessError(errno.ECHILD, "simulated competing reaper")
    with monkeypatch.context() as patch:
        patch.setattr(os, "waitpid", stolen)
        assert owner.finish() == "lost"
    assert owner.retry() == "lost"


def test_final_group_kill_stops_term_ignoring_descendant(tmp_path):
    heartbeat = tmp_path / "heartbeat"
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", """
import os, pathlib, signal, sys, time
if os.fork() == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        pathlib.Path(sys.argv[1]).write_text(str(time.monotonic_ns()))
        time.sleep(0.01)
else:
    sys.stdin.buffer.read()
""", str(heartbeat)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    owner = ChildCleanup(process)
    try:
        deadline = time.monotonic() + 3
        while not heartbeat.exists():
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert owner.finish() == "complete"
        first = heartbeat.read_bytes()
        time.sleep(0.1)
        assert heartbeat.read_bytes() == first
        assert process.returncode is not None
    finally:
        process.stdin.close()
        owner.retry()
