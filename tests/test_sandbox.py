"""Executor boundary tests with real bubblewrap and installed runtime closures.

Failure modes: whole-store/devshell exposure; writable spec aliases; inherited
credentials/fds; host file/process/network access; output floods; cancellation,
timeout and child leaks. These tests do not qualify any coding-agent context.
"""
from collections import UserList
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

from r2spec import _child, sandbox as sandbox_module
from r2spec.sandbox import Sandbox, SandboxError

PYTHON_ROOT = Path(os.__file__).resolve().parents[2]
PYTHON = PYTHON_ROOT / "bin/python3"


@pytest.fixture
def directories(tmp_path):
    specs = tmp_path / "specs"
    work = tmp_path / "work"
    specs.mkdir()
    work.mkdir()
    (specs / "spec.json").write_text('{"approved_test_vector": 4}\n')
    return specs, work


@pytest.fixture
def sandbox():
    return Sandbox([PYTHON_ROOT])


def child_pids():
    return Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()


def test_approved_spec_read_and_workspace_write(sandbox, directories):
    specs, work = directories
    script = """
import json, pathlib
spec = json.loads(pathlib.Path("/specs/spec.json").read_text())
pathlib.Path("/work/result.txt").write_text(str(spec["approved_test_vector"]))
print(json.dumps({"spec": spec["approved_test_vector"], "written": True}))
"""
    result = sandbox.run([str(PYTHON), "-I", "-c", script], specs=specs, work=work)
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"spec": 4, "written": True}
    assert (work / "result.txt").read_text() == "4"


def test_real_file_process_network_env_and_fd_denials(sandbox, directories, tmp_path, monkeypatch):
    specs, work = directories
    secret = tmp_path / "inspector-only.txt"
    secret.write_text("synthetic original information\n")
    (specs / "outside-link").symlink_to(secret)
    before = secret.read_bytes()
    before_spec = (specs / "spec.json").read_bytes()
    monkeypatch.setenv("R2SPEC_PRIVATE_CANARY", "synthetic credential")
    process = subprocess.Popen(
        [str(PYTHON), "-I", "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    fd = os.open(secret, os.O_RDONLY)
    inherited_fd = fcntl.fcntl(fd, fcntl.F_DUPFD, 200)
    os.set_inheritable(inherited_fd, True)
    script = """
import json, os, pathlib, socket, sys
secret, pid, port, fd, r2 = sys.argv[1:]
checks = {}
for name, path in (("original", secret), ("symlink", "/specs/outside-link")):
    try:
        pathlib.Path(path).read_bytes()
        checks[name + "_read_denied"] = False
    except OSError:
        checks[name + "_read_denied"] = True
for name, path in (("original", secret), ("spec", "/specs/spec.json")):
    try:
        pathlib.Path(path).write_text("forbidden mutation")
        checks[name + "_write_denied"] = False
    except OSError:
        checks[name + "_write_denied"] = True
checks["host_process_hidden"] = not pathlib.Path("/proc", pid).exists()
try:
    os.kill(int(pid), 0)
    checks["host_signal_denied"] = False
except (ProcessLookupError, PermissionError):
    checks["host_signal_denied"] = True
with socket.socket() as connection:
    connection.settimeout(0.3)
    try:
        connection.connect(("127.0.0.1", int(port)))
        checks["host_loopback_denied"] = False
    except OSError:
        checks["host_loopback_denied"] = True
try:
    os.fstat(int(fd))
    checks["inherited_fd_closed"] = False
except OSError:
    checks["inherited_fd_closed"] = True
checks["parent_env_absent"] = "R2SPEC_PRIVATE_CANARY" not in os.environ
checks["r2_direct_path_absent"] = not pathlib.Path(r2).exists()
print(json.dumps(checks))
"""
    try:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(0.1)
            result = sandbox.run(
                [str(PYTHON), "-I", "-c", script, str(secret), str(process.pid),
                 str(listener.getsockname()[1]), str(inherited_fd), shutil.which("r2")],
                specs=specs, work=work,
            )
            assert result.returncode == 0, result.stderr
            results = json.loads(result.stdout)
            assert results and all(results.values()), results
            with pytest.raises(TimeoutError):
                connection, _ = listener.accept()
                connection.close()
        assert secret.read_bytes() == before
        assert (specs / "spec.json").read_bytes() == before_spec
        assert process.poll() is None
    finally:
        os.close(inherited_fd)
        os.close(fd)
        process.terminate()
        process.wait(timeout=3)


def test_rust_metadata_compilation_without_target_execution(directories):
    specs, work = directories
    rustc = Path(shutil.which("rustc"))
    executor = Sandbox([rustc.parents[1]])
    (work / "lib.rs").write_text("pub fn fixture_sum(a: i16, b: i16) -> i32 { a as i32 + b as i32 }\n")
    result = executor.run(
        [str(rustc), "--emit=metadata", "--crate-type=lib", "/work/lib.rs",
         "-o", "/work/lib.rmeta"], specs=specs, work=work, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert (work / "lib.rmeta").stat().st_size > 0


def test_nonzero_child_exit_is_reported(sandbox, directories):
    specs, work = directories
    result = sandbox.run(
        [str(PYTHON), "-I", "-c", "import sys; print('failure', file=sys.stderr); sys.exit(7)"],
        specs=specs, work=work,
    )
    assert result.returncode == 7
    assert result.stderr == b"failure\n"


@pytest.mark.parametrize("channel", [1, 2])
def test_output_cap_is_enforced_during_reception(sandbox, directories, channel):
    specs, work = directories
    before = child_pids()
    with pytest.raises(SandboxError) as error:
        sandbox.run(
            [str(PYTHON), "-I", "-c",
             f"import os; data=b'x'*65536\nwhile True: os.write({channel},data)"],
            specs=specs, work=work, max_output_bytes=1024,
        )
    assert error.value.code == "overflow"
    assert child_pids() == before


def test_timeout_kills_namespace_and_descendants(sandbox, directories):
    specs, work = directories
    before = child_pids()
    script = """
import os, pathlib, signal, time
if os.fork() == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        pathlib.Path("/work/heartbeat").write_text(str(time.monotonic_ns()))
        time.sleep(0.01)
else:
    time.sleep(60)
"""
    with pytest.raises(SandboxError) as error:
        sandbox.run(
            [str(PYTHON), "-I", "-c", script],
            specs=specs, work=work, timeout=0.5,
        )
    assert error.value.code == "timeout"
    assert child_pids() == before
    first = (work / "heartbeat").read_bytes()
    time.sleep(0.15)
    assert (work / "heartbeat").read_bytes() == first


def test_cancellation_is_bounded(sandbox, directories):
    specs, work = directories
    cancel = threading.Event()
    timer = threading.Timer(0.25, cancel.set)
    before = child_pids()
    timer.start()
    try:
        with pytest.raises(SandboxError) as error:
            sandbox.run(
                [str(PYTHON), "-I", "-c", "import time; time.sleep(60)"],
                specs=specs, work=work, cancel=cancel,
            )
        assert error.value.code == "cancelled"
    finally:
        timer.cancel()
        timer.join()
    assert child_pids() == before


@pytest.mark.parametrize("roots", [
    [], [Path("/nix/store")], [Path("/")], [Path(sys.prefix)],
])
def test_broad_or_r2_containing_runtime_is_rejected(roots):
    with pytest.raises(SandboxError) as error:
        Sandbox(roots)
    assert error.value.code == "input"


def test_canonical_installed_r2_root_is_rejected():
    executable = shutil.which("r2")
    assert executable is not None, "qualified environment requires installed r2"
    root = Path(executable).resolve(strict=True).parents[1]
    assert root.parent == Path("/nix/store")
    with pytest.raises(SandboxError) as error:
        Sandbox([root])
    assert error.value.code == "input"


def test_direct_r2_command_is_rejected(sandbox, directories):
    specs, work = directories
    with pytest.raises(SandboxError) as error:
        sandbox.run([shutil.which("r2"), "-v"], specs=specs, work=work)
    assert error.value.code == "input"


@pytest.mark.parametrize("relationship", ["same", "nested", "symlink"])
def test_ambiguous_or_overlapping_mounts_are_rejected(sandbox, directories, tmp_path, relationship):
    specs, work = directories
    if relationship == "same":
        work = specs
    elif relationship == "nested":
        work = specs / "nested"
        work.mkdir()
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(specs, target_is_directory=True)
        specs = alias
    with pytest.raises(SandboxError) as error:
        sandbox.run([str(PYTHON), "--version"], specs=specs, work=work)
    assert error.value.code == "input"


@pytest.mark.parametrize("options", [
    {"timeout": 0}, {"timeout": float("nan")}, {"timeout": True},
    {"max_output_bytes": 0}, {"max_output_bytes": True},
])
def test_invalid_run_limits(sandbox, directories, options):
    specs, work = directories
    with pytest.raises(SandboxError) as error:
        sandbox.run([str(PYTHON), "--version"], specs=specs, work=work, **options)
    assert error.value.code == "input"


def test_missing_bubblewrap_fails_closed(tmp_path):
    with pytest.raises(SandboxError) as error:
        Sandbox([PYTHON_ROOT], executable=tmp_path / "missing")
    assert error.value.code == "unavailable"


# Cleanup failure modes mirror the transport: never replace a primary, publish a
# result before cleanup, or reap the group identity after failed final signaling.
@pytest.mark.parametrize("mode", ["success", "timeout", "overflow"])
def test_sandbox_cleanup_failure_exposes_owner(sandbox, directories, monkeypatch, mode):
    specs, work = directories
    real_popen, real_killpg = subprocess.Popen, os.killpg
    spawned = []
    def record_spawn(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        spawned.append(process)
        return process
    def deny_signal(*args):
        raise PermissionError("simulated group-signal refusal")
    script = {
        "success": "import sys; print('done'); sys.exit(7)",
        "timeout": "import time; time.sleep(60)",
        "overflow": "import os,time; os.write(1,b'x'*8192); time.sleep(60)",
    }[mode]
    try:
        with monkeypatch.context() as patch:
            patch.setattr(subprocess, "Popen", record_spawn)
            patch.setattr(_child, "_signal_group", deny_signal)
            with pytest.raises(SandboxError) as caught:
                sandbox.run(
                    [str(PYTHON), "-I", "-c", script], specs=specs, work=work,
                    timeout=0.3, max_output_bytes=64,
                )
            assert caught.value.code == ("cleanup" if mode == "success" else mode)
            owner = caught.value.cleanup
            assert owner is not None and owner.state == "pending"
            assert caught.value.__notes__
            assert spawned[0].returncode is None
            os.waitid(os.P_PID, spawned[0].pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        assert owner.retry() == "complete"
        if mode == "success":
            assert owner.returncode == 7
        assert not Path(f"/proc/{spawned[0].pid}").exists()
    finally:
        for process in spawned:
            try:
                os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            except ChildProcessError:
                continue
            try:
                real_killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
            for stream in (process.stdout, process.stderr):
                stream.close()


# Setup failure isolation: owner allocation after spawn can lose the child;
# post-binding setup can mask a primary or lose pending cleanup; failed exec can
# reach teardown without a handle. Inject allocation/setup failures, not actual
# exhaustion. Spawn, signals, exit observation and reaping remain real.
@pytest.fixture
def sandbox_spawns(sandbox, monkeypatch):
    real_popen, real_killpg = subprocess.Popen, os.killpg
    spawned = []
    def record_spawn(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        spawned.append(process)
        return process
    monkeypatch.setattr(subprocess, "Popen", record_spawn)
    yield spawned
    for process in spawned:
        try:
            os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            pass
        else:
            try:
                real_killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
        for stream in (process.stdout, process.stderr):
            stream.close()


def test_sandbox_owner_allocation_failure_precedes_spawn(
    sandbox, directories, sandbox_spawns, monkeypatch,
):
    primary = MemoryError("injected owner allocation failure, not real exhaustion")
    calls = []
    def failed_owner(*args, **kwargs):
        calls.append(True)
        raise primary
    monkeypatch.setattr(sandbox_module, "ChildCleanup", failed_owner)
    specs, work = directories
    with pytest.raises(MemoryError) as caught:
        sandbox.run(
            [str(PYTHON), "-I", "-c", "import time; time.sleep(60)"],
            specs=specs, work=work,
        )
    assert caught.value is primary
    assert getattr(primary, "cleanup", None) is None
    assert calls == [True]
    assert sandbox_spawns == [], "child spawned before cleanup owner existed"


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_sandbox_post_binding_setup_preserves_primary_and_owner(
    sandbox, directories, sandbox_spawns, monkeypatch, pending, error_type,
):
    primary = error_type("injected selector setup failure")
    def failed_setup():
        raise primary
    def denied(*args):
        raise PermissionError("injected group-signal refusal")
    specs, work = directories
    with monkeypatch.context() as patch:
        patch.setattr(sandbox_module.selectors, "DefaultSelector", failed_setup)
        if pending:
            patch.setattr(_child, "_signal_group", denied)
        with pytest.raises(error_type) as caught:
            sandbox.run(
                [str(PYTHON), "-I", "-c", "import time; time.sleep(60)"],
                specs=specs, work=work,
            )
        assert caught.value is primary
        assert len(sandbox_spawns) == 1
        process = sandbox_spawns[0]
        assert process.stdout.closed and process.stderr.closed
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


def test_sandbox_exec_failure_preserves_unavailable_without_handle(directories, tmp_path):
    broken = tmp_path / "broken-bwrap"
    broken.write_text(f"#!{tmp_path / 'missing-interpreter'}\n")
    broken.chmod(0o700)
    executor = Sandbox([PYTHON_ROOT], executable=broken)
    specs, work = directories
    before_children = child_pids()
    before_fds = len(os.listdir("/proc/self/fd"))
    with pytest.raises(SandboxError) as caught:
        executor.run([str(PYTHON), "-I", "-c", "pass"], specs=specs, work=work)
    assert caught.value.code == "unavailable"
    assert isinstance(caught.value.__cause__, FileNotFoundError)
    assert caught.value.cleanup is None
    assert child_pids() == before_children
    assert len(os.listdir("/proc/self/fd")) == before_fds


def test_sandbox_signals_precede_reap(sandbox, directories, monkeypatch):
    specs, work = directories
    real_killpg = os.killpg
    signals = []
    def checked_signal(pid, sig):
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        signals.append(sig)
        return real_killpg(pid, sig)
    with monkeypatch.context() as patch:
        patch.setattr(_child, "_signal_group", checked_signal)
        result = sandbox.run([str(PYTHON), "-I", "-c", "pass"], specs=specs, work=work)
    assert result.returncode == 0
    assert signals[-1] == signal.SIGKILL


def test_sandbox_detects_external_reap_without_signaling(sandbox, directories, monkeypatch):
    specs, work = directories
    real_popen = subprocess.Popen
    def externally_reaped(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        process.wait(timeout=3)  # Deliberately violate the embedding precondition.
        return process
    def forbidden_signal(*args):
        pytest.fail("signaled a group whose leader was already reaped")
    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "Popen", externally_reaped)
        patch.setattr(_child, "_signal_group", forbidden_signal)
        with pytest.raises(SandboxError) as error:
            sandbox.run([str(PYTHON), "-I", "-c", "pass"], specs=specs, work=work)
        assert error.value.code == "process"
        assert error.value.cleanup.state == "lost"
        assert error.value.cleanup.retry() == "lost"


@pytest.mark.parametrize("directory", [
    "/", "/nix", "/nix/var", "/proc", "/sys", "/dev", "/dev/shm", "/run",
])
@pytest.mark.parametrize("role", ["specs", "work"])
def test_host_tree_mounts_rejected_before_spawn(sandbox, directories, monkeypatch, directory, role):
    specs, work = directories
    def forbidden_spawn(*args, **kwargs):
        pytest.fail("unsafe directory validation reached process creation")
    monkeypatch.setattr(subprocess, "Popen", forbidden_spawn)
    paths = {"specs": specs, "work": work}
    paths[role] = directory
    with pytest.raises(SandboxError) as error:
        sandbox.run([str(PYTHON), "--version"], **paths)
    assert error.value.code == "input"


def test_runtime_and_spec_mount_flags_are_read_only(sandbox, directories):
    specs, work = directories
    # Inspect actual mount flags without ever writing an installed store file.
    script = """
import json, pathlib
mounts = {}
for line in pathlib.Path("/proc/self/mountinfo").read_text().splitlines():
    fields = line.split()
    mounts[fields[4]] = fields[5].split(",")
print(json.dumps(mounts))
"""
    result = sandbox.run(
        [str(PYTHON), "-I", "-c", script], specs=specs, work=work,
    )
    assert result.returncode == 0
    mounts = json.loads(result.stdout)
    roots = subprocess.run(
        ["nix-store", "--query", "--requisites", str(PYTHON_ROOT)],
        check=True, capture_output=True, text=True, timeout=5,
    ).stdout.splitlines()
    assert roots
    for path in [*roots, "/specs"]:
        assert "ro" in mounts[path], path
        assert "rw" not in mounts[path], path
    assert "rw" in mounts["/work"]


@pytest.mark.parametrize("command", [
    "", "python", b"python", [], [""], ["python"], ["bin/python"],
    [str(PYTHON), "\0"], [None], [str(PYTHON), 7],
])
def test_malformed_commands_rejected_without_spawn(sandbox, directories, monkeypatch, command):
    specs, work = directories
    def forbidden_spawn(*args, **kwargs):
        pytest.fail("malformed command reached process creation")
    monkeypatch.setattr(subprocess, "Popen", forbidden_spawn)
    with pytest.raises(SandboxError) as error:
        sandbox.run(command, specs=specs, work=work)
    assert error.value.code == "input"


# Container failures: dictionaries can be iterated as roots/arguments, generators
# can be consumed before indexing, and an empty iterator can request zero roots.
# Reject the outer shape before querying/spawning, without consuming iterables.
@pytest.mark.parametrize("kind", [
    "dict", "generator", "empty-generator", "iterator", "set",
    "path", "str", "bytes", "bytearray", "int", "none",
])
@pytest.mark.parametrize("boundary", ["roots", "command"])
def test_non_sequence_containers_rejected_before_spawn(
    sandbox, directories, monkeypatch, kind, boundary,
):
    values = [str(PYTHON_ROOT)] if boundary == "roots" else [str(PYTHON), "--version"]
    containers = {
        "dict": lambda: dict.fromkeys(values),
        "generator": lambda: (value for value in values),
        "empty-generator": lambda: (value for value in ()),
        "iterator": lambda: iter(values),
        "set": lambda: set(values),
        "path": lambda: Path(values[0]),
        "str": lambda: values[0],
        "bytes": lambda: values[0].encode(),
        "bytearray": lambda: bytearray(values[0].encode()),
        "int": lambda: 7,
        "none": lambda: None,
    }
    value = containers[kind]()
    def forbidden_spawn(*args, **kwargs):
        pytest.fail("invalid container reached query or process creation")
    monkeypatch.setattr(subprocess, "Popen", forbidden_spawn)
    with pytest.raises(SandboxError) as error:
        if boundary == "roots":
            Sandbox(value)
        else:
            specs, work = directories
            sandbox.run(value, specs=specs, work=work)
    assert error.value.code == "input"
    if kind in ("generator", "iterator"):
        assert list(value) == values, "validation consumed a rejected container"


@pytest.mark.parametrize("container", [list, tuple, UserList])
def test_valid_sequence_containers_preserve_execution(directories, container):
    specs, work = directories
    executor = Sandbox(container([PYTHON_ROOT]))
    result = executor.run(
        container([str(PYTHON), "-I", "-c", "print('sequence-ok')"]),
        specs=specs, work=work,
    )
    assert result.returncode == 0
    assert result.stdout == b"sequence-ok\n"


def test_runtime_readonly_independent_of_host_store(directories, tmp_path):
    specs, work = directories
    bwrap = Path(shutil.which("bwrap")).resolve()
    extra_runtime = bwrap.parents[1]
    scratch = tmp_path / "scratch-runtime"
    scratch.mkdir()
    (scratch / "fixture-token").write_text("scratch-only")
    launcher = tmp_path / "bwrap-fixture"
    # Adapt only an unused runtime's source at the external bwrap boundary.
    # Keep the production bind option and destination. Real bwrap mounts a
    # writable-host scratch tree, so inherited store flags cannot mask --bind.
    # Abort before launching if the exact substitution is missing/ambiguous.
    launcher.write_text(f"""#!{sys.executable}
import os, sys
args = sys.argv[1:]
matches = [i for i in range(len(args)-2)
           if args[i] in ("--ro-bind", "--bind")
           and args[i+1:i+3] == [{str(extra_runtime)!r}, {str(extra_runtime)!r}]]
if len(matches) != 1: sys.exit(99)
args[matches[0]+1] = {str(scratch)!r}
os.execv({str(bwrap)!r}, [{str(bwrap)!r}, *args])
""")
    launcher.chmod(0o700)
    executor = Sandbox([PYTHON_ROOT, extra_runtime], executable=launcher)
    script = """
import json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
assert (root / "fixture-token").read_text() == "scratch-only"
print(json.dumps(bool(os.statvfs(root).f_flag & os.ST_RDONLY)))
"""
    result = executor.run(
        [str(PYTHON), "-I", "-c", script, str(extra_runtime)],
        specs=specs, work=work,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) is True
    assert (scratch / "fixture-token").read_text() == "scratch-only"
