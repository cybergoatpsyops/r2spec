"""Real ``python -m r2spec`` process tests for mode routing, RE inspect and the ledger.

Failure modes: help or refusal importing transport, starting r2 or writing
files; malformed arguments reaching r2 or the ledger; evidence published after
transport, target, header, write or cleanup failure or interruption;
non-private or non-repeatable evidence; missing r2 breaking help or refusal; a
repeated interrupt breaking child cleanup; a ledger route loading the RE
stack, mapping a refusal, failure or interruption to the wrong status,
printing raw control characters or failing on a closed stdout after its
commit. Fake executables substitute only for r2. In-process checks inject only
OS failures or signal timing that cannot be induced on demand; they prove exit
mapping and rollback, not containment.
"""
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import stat
import subprocess
import sys
import textwrap
import threading
import time
from types import SimpleNamespace

import pytest

from r2spec import _child, cli

REPO = Path(__file__).resolve().parents[1]
NATIVE = Path(__file__).parent / "fixtures/native.c"
RE_STACK = {"r2spec.transport", "r2spec._child", "r2spec.re_inspect"}
REFUSAL = (
    "r2spec: refused: clean-room mode is not qualified in this build; "
    "no specification was read and no artifact was written."
)

_FAKE_HELPERS = textwrap.dedent("""
    import json, os, signal, sys, time
    from pathlib import Path
    target = sys.argv[-1]

    def reply(data):
        sys.stdout.buffer.write(data + b"\\0")
        sys.stdout.flush()

    def metadata():
        core = {"file": target, "size": os.path.getsize(target), "format": "null"}
        return json.dumps({"core": core}).encode()

    def serve():
        while line := sys.stdin.buffer.readline():
            if line.startswith(b"ij"):
                reply(metadata())
            else:
                count = int(line.split()[1])
                reply(json.dumps(list(Path(target).read_bytes()[:count])).encode())
""")
GOOD = 'reply(b""); serve()'


@pytest.fixture
def dirs(tmp_path):
    for name in ("home", "cwd", "targets", "outputs"):
        (tmp_path / name).mkdir()
    sample = tmp_path / "targets" / "sample.bin"
    sample.write_bytes(bytes(range(32)))
    return SimpleNamespace(
        root=tmp_path, home=tmp_path / "home", cwd=tmp_path / "cwd",
        targets=tmp_path / "targets", outputs=tmp_path / "outputs", sample=sample,
    )


def fake_r2(dirs, body):
    """Install a fake r2 alone on a PATH directory; it records its pid when launched."""
    directory = dirs.root / "fake-bin"
    directory.mkdir()
    pid_file = dirs.root / "fake-r2.pid"
    executable = directory / "r2"
    executable.write_text(
        f"#!{sys.executable}\n" + _FAKE_HELPERS
        + f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        + textwrap.dedent(body) + "\n"
    )
    executable.chmod(0o700)
    return str(directory), pid_file


def empty_path(dirs):
    directory = dirs.root / "empty-bin"
    directory.mkdir()
    return str(directory)


def invocation(dirs, args, path=None, importtime=False):
    command = [sys.executable, *(["-X", "importtime"] if importtime else []), "-m", "r2spec"]
    env = {
        "PATH": os.environ["PATH"] if path is None else path,
        "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": str(dirs.home), "LC_ALL": "C",
    }
    return [*command, *map(str, args)], env


def run_cli(dirs, *args, path=None, importtime=False, **options):
    command, env = invocation(dirs, args, path, importtime)
    return subprocess.run(
        command, env=env, cwd=dirs.cwd, capture_output=True, text=True, timeout=30, **options,
    )


def imported(stderr):
    return {
        line.rsplit("|", 1)[1].strip()
        for line in stderr.splitlines() if line.startswith("import time:")
    }


def messages(stderr):
    return [line for line in stderr.splitlines() if line.startswith("r2spec:")]


def snapshot(*roots):
    state = {}
    for root in roots:
        for path in [root, *sorted(root.rglob("*"))]:
            info = path.lstat()
            data = path.read_bytes() if stat.S_ISREG(info.st_mode) else None
            state[str(path)] = (info.st_mode, data)
    return state


def assert_gone(pid_file):
    assert pid_file.exists(), "fake r2 never started"
    assert not Path("/proc", pid_file.read_text().strip()).exists()


def assert_nothing_published(dirs, result=None):
    assert list(dirs.outputs.iterdir()) == []
    if result is not None:
        assert result.stdout == ""
        assert "Traceback" not in result.stderr


def build_native(dirs):
    assert platform.machine() == "aarch64"
    target = dirs.targets / "native"
    subprocess.run(
        ["cc", "-g", "-O0", str(NATIVE), "-o", str(target)],
        check=True, capture_output=True, timeout=20,
    )
    target.chmod(0o600)  # Not executable: inspection cannot have run it.
    return target


def published_path(out):
    return out.parent.resolve() / out.name / "evidence.json"


# Mode routing: no mode, help and refusal must stay free of the RE stack.
def test_no_mode_prints_mode_help_and_exits_2_without_side_effects(dirs):
    fake_path, pid_file = fake_r2(dirs, GOOD)
    before = snapshot(dirs.home, dirs.cwd, dirs.targets, dirs.outputs)
    result = run_cli(dirs, path=fake_path, importtime=True)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "usage: python -m r2spec" in result.stderr
    assert "clean-room" in result.stderr and "re " in result.stderr
    assert "r2spec.cli" in imported(result.stderr)
    assert not imported(result.stderr) & RE_STACK
    assert not pid_file.exists()
    assert snapshot(dirs.home, dirs.cwd, dirs.targets, dirs.outputs) == before


@pytest.mark.parametrize("args", [
    ["-h"], ["re", "-h"], ["re", "inspect", "-h"], ["clean-room", "-h"], ["ledger", "-h"], ["ledger", "close", "-h"],
])
def test_help_exits_0(dirs, args):
    result = run_cli(dirs, *args, path=empty_path(dirs), importtime=True)
    assert result.returncode == 0
    assert result.stdout.startswith("usage: python -m r2spec")
    assert not imported(result.stderr) & RE_STACK


@pytest.mark.parametrize("args", [
    ["bogus"], ["re"], ["re", "bogus"], ["re", "inspect"], ["re", "inspect", "t"],
    ["re", "inspect", "--out", "d"], ["re", "insp", "t", "--out", "d"],
    ["re", "inspect", "t", "--ou", "d"], ["re", "inspect", "t", "--out", "d", "--extra"],
    ["clean-room", "extra"], ["clean-room", "--out", "d"],
    ["ledger"], ["ledger", "bogus"], ["ledger", "add", "a", "--requirement", "r"],
    ["ledger", "add", "a", "--requirement", "r", "--ledger", "d", "--expect-rev", "0"],
    ["ledger", "show", "--ledger", "d", "--expect-revision", "1"],
    ["ledger", "close", "a", "--ledger", "d", "--expect-revision", "1", "--accept", "--rationale", "r"],
    ["ledger", "close", "a", "--ledger", "d", "--expect-revision", "1", "--rationale", "r", "--no-known-limitations"],
])
def test_malformed_args_exit_2_without_starting_r2(dirs, args):
    fake_path, pid_file = fake_r2(dirs, GOOD)
    (dirs.cwd / "t").write_bytes(b"target")
    before = snapshot(dirs.cwd, dirs.home)
    result = run_cli(dirs, *args, path=fake_path, importtime=True)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "usage:" in result.stderr
    assert not imported(result.stderr) & RE_STACK
    assert not pid_file.exists()
    assert snapshot(dirs.cwd, dirs.home) == before


def test_clean_room_refuses_unqualified_exit_3_without_r2(dirs):
    before = snapshot(dirs.home, dirs.cwd, dirs.targets, dirs.outputs)
    result = run_cli(dirs, "clean-room", path=empty_path(dirs))
    assert result.returncode == 3
    assert result.stdout == ""
    assert messages(result.stderr) == [REFUSAL]
    assert "complete" not in result.stderr and "success" not in result.stderr
    assert snapshot(dirs.home, dirs.cwd, dirs.targets, dirs.outputs) == before


def test_clean_room_never_imports_transport_or_starts_r2(dirs):
    fake_path, pid_file = fake_r2(dirs, GOOD)
    result = run_cli(dirs, "clean-room", path=fake_path, importtime=True)
    assert result.returncode == 3
    assert "r2spec.cli" in imported(result.stderr)
    assert not imported(result.stderr) & RE_STACK
    assert not pid_file.exists()


def test_import_probe_detects_transport_on_re_route(dirs):
    out = dirs.outputs / "evidence"
    result = run_cli(
        dirs, "re", "inspect", dirs.sample, "--out", out, path=empty_path(dirs), importtime=True,
    )
    assert result.returncode == 4
    assert {"r2spec.re_inspect", "r2spec.transport"} <= imported(result.stderr)


# RE inspect success with the real radare2 and a compiled, never-executed fixture.
def test_re_inspect_native_fixture_publishes_private_evidence(dirs):
    target = build_native(dirs)
    content = target.read_bytes()
    out = dirs.outputs / "evidence"
    before = snapshot(dirs.home, dirs.cwd, dirs.targets)
    result = run_cli(dirs, "re", "inspect", target, "--out", out)
    assert result.returncode == 0, result.stderr
    published = published_path(out)
    assert result.stdout == f"{published}\n"
    assert snapshot(dirs.home, dirs.cwd, dirs.targets) == before
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    assert [path.name for path in out.iterdir()] == ["evidence.json"]
    assert stat.S_IMODE(published.stat().st_mode) == 0o600
    evidence = json.loads(published.read_text())
    assert evidence["schema"] == "r2spec.re.inspect/1"
    assert (evidence["mode"], evidence["operation"]) == ("re", "inspect")
    assert evidence["target"] == {
        "path": str(target.resolve()), "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(), "confidence": "observed",
    }
    assert evidence["collection"]["operations"] == [
        {"op": "metadata", "r2": "ij"}, {"op": "read_bytes", "offset": 0, "count": 64},
    ]
    assert evidence["collection"]["trust"] == {"r2_sandboxed": False, "replies_authenticated": False}
    facts = evidence["facts"]
    assert facts["format"] == {"value": "elf64", "confidence": "reported", "source": "ij.core.format"}
    assert facts["arch"]["value"] == "arm"
    assert facts["bits"]["value"] == 64
    assert facts["header_bytes"] == {
        "value": content[:64].hex(), "offset": 0, "length": 64,
        "confidence": "corroborated", "source": "pxj",
    }
    assert facts["header_bytes"]["value"].startswith("7f454c46")
    assert {fact["confidence"] for fact in facts.values()} <= {"observed", "corroborated", "reported"}
    assert {"field": "tool.r2_version", "reason": "not_exposed"} in evidence["unknowns"]
    assert evidence["unknowns"] == sorted(evidence["unknowns"], key=lambda item: item["field"])


def test_re_inspect_evidence_is_byte_repeatable(dirs):
    target = build_native(dirs)
    first, second = dirs.outputs / "first", dirs.outputs / "second"
    assert run_cli(dirs, "re", "inspect", target, "--out", first).returncode == 0
    assert run_cli(dirs, "re", "inspect", target, "--out", second).returncode == 0
    assert published_path(first).read_bytes() == published_path(second).read_bytes()


def test_re_inspect_empty_target_records_unknowns(dirs):
    target = dirs.targets / "empty.bin"
    target.write_bytes(b"")
    out = dirs.outputs / "evidence"
    result = run_cli(dirs, "re", "inspect", target, "--out", out)
    assert result.returncode == 0, result.stderr
    evidence = json.loads(published_path(out).read_text())
    assert evidence["collection"]["operations"] == [{"op": "metadata", "r2": "ij"}]
    assert evidence["facts"] == {}
    assert {"field": "facts.header_bytes", "reason": "empty_target"} in evidence["unknowns"]
    assert {"field": "facts.format", "reason": "unrecognized"} in evidence["unknowns"]
    assert {"field": "facts.arch", "reason": "absent"} in evidence["unknowns"]


# Invalid invocation is refused before r2 starts and creates nothing.
@pytest.mark.parametrize("kind", ["directory", "file", "dangling symlink"])
def test_re_inspect_existing_out_refused_before_r2(dirs, kind):
    fake_path, pid_file = fake_r2(dirs, GOOD)
    out = dirs.outputs / "evidence"
    if kind == "directory":
        out.mkdir()
    elif kind == "file":
        out.write_text("keep")
    else:
        out.symlink_to(dirs.outputs / "missing")
    before = snapshot(dirs.outputs)
    result = run_cli(dirs, "re", "inspect", dirs.sample, "--out", out, path=fake_path)
    assert result.returncode == 2
    assert messages(result.stderr) == ["r2spec: invocation: --out already exists; name a new directory"]
    assert snapshot(dirs.outputs) == before
    assert not pid_file.exists()


@pytest.mark.parametrize("case", ["missing target", "directory target", "missing out parent", "file out parent"])
def test_re_inspect_invalid_target_or_out_parent_exit_2(dirs, case):
    fake_path, pid_file = fake_r2(dirs, GOOD)
    target, out = dirs.sample, dirs.outputs / "evidence"
    if case == "missing target":
        target = dirs.targets / "absent"
    elif case == "directory target":
        target = dirs.targets
    elif case == "missing out parent":
        out = dirs.outputs / "absent" / "evidence"
    else:
        out = dirs.sample / "evidence"
    before = snapshot(dirs.targets, dirs.outputs)
    result = run_cli(dirs, "re", "inspect", target, "--out", out, path=fake_path)
    assert result.returncode == 2
    assert [line.split(":")[1] for line in messages(result.stderr)] == [" invocation"]
    assert snapshot(dirs.targets, dirs.outputs) == before
    assert not pid_file.exists()


def test_re_inspect_unstable_size_target_refused_accurately(dirs):
    fake_path, pid_file = fake_r2(dirs, GOOD)
    result = run_cli(
        dirs, "re", "inspect", "/proc/version", "--out", dirs.outputs / "evidence", path=fake_path,
    )
    assert result.returncode == 1
    assert messages(result.stderr) == [
        "r2spec: failed: TARGET size does not match the bytes read; "
        "it is not a stable regular file; no evidence was published",
    ]
    assert_nothing_published(dirs, result)
    assert not pid_file.exists()


def test_re_inspect_missing_r2_exit_4_publishes_nothing(dirs):
    result = run_cli(
        dirs, "re", "inspect", dirs.sample, "--out", dirs.outputs / "evidence", path=empty_path(dirs),
    )
    assert result.returncode == 4
    assert messages(result.stderr) == [
        "r2spec: unavailable: radare2 executable is unavailable; no evidence was published",
    ]
    assert_nothing_published(dirs, result)


# Transport, target and corroboration failures never publish partial evidence.
@pytest.mark.parametrize("body,reason", [
    ('reply(b""); sys.stdin.buffer.readline(); sys.exit(4)', "radare2 process"),
    ('reply(b""); sys.stdin.buffer.readline(); reply(b"not json"); time.sleep(30)', "radare2 json"),
    ('reply(b""); sys.stdin.buffer.readline()\n'
     'reply(json.dumps({"core": {"file": "/elsewhere", "size": 32}}).encode()); time.sleep(30)',
     "radare2 schema"),
    ('reply(b""); sys.stdin.buffer.readline(); reply(metadata())\n'
     'count = int(sys.stdin.buffer.readline().split()[1])\n'
     'reply(json.dumps([255] * count).encode()); serve()',
     "radare2 header bytes disagree with the local read"),
    ('reply(b""); sys.stdin.buffer.readline()\n'
     'Path(target).write_bytes(bytes(reversed(Path(target).read_bytes())))\n'
     'reply(metadata()); serve()',
     "radare2 source_changed"),
], ids=["peer exits", "garbage frame", "wrong file", "header mismatch", "target rewritten"])
def test_re_inspect_failure_publishes_nothing(dirs, body, reason):
    fake_path, pid_file = fake_r2(dirs, body)
    result = run_cli(dirs, "re", "inspect", dirs.sample, "--out", dirs.outputs / "evidence", path=fake_path)
    assert result.returncode == 1
    [message] = messages(result.stderr)
    assert message.startswith(f"r2spec: failed: {reason}")
    assert message.endswith("no evidence was published")
    assert_nothing_published(dirs, result)
    assert_gone(pid_file)


# Interruption: the CLI reports 130 only after reaping r2 and removing its output.
def interrupt(dirs, signals, ignore_term=False):
    ready = dirs.root / "ready"
    body = ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "") + (
        f'reply(b""); sys.stdin.buffer.readline(); Path({str(ready)!r}).write_text("1"); time.sleep(30)'
    )
    fake_path, pid_file = fake_r2(dirs, body)
    command, env = invocation(dirs, ["re", "inspect", dirs.sample, "--out", dirs.outputs / "evidence"], fake_path)
    process = subprocess.Popen(
        command, env=env, cwd=dirs.cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists():
            assert time.monotonic() < deadline, "fake r2 never received a request"
            assert process.poll() is None, process.stderr.read()
            time.sleep(0.01)
        for _ in range(signals):
            process.send_signal(signal.SIGINT)
            time.sleep(0.05)
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    return SimpleNamespace(returncode=process.returncode, stdout=stdout, stderr=stderr), pid_file


def test_re_inspect_sigint_during_exchange_exits_130_and_reaps(dirs):
    result, pid_file = interrupt(dirs, signals=1)
    assert result.returncode == 130
    assert messages(result.stderr) == ["r2spec: interrupted: interrupted; no evidence was published"]
    assert_nothing_published(dirs, result)
    assert_gone(pid_file)


def test_re_inspect_repeated_sigint_still_cleans_up(dirs):
    result, pid_file = interrupt(dirs, signals=3, ignore_term=True)
    assert result.returncode == 130
    assert messages(result.stderr) == ["r2spec: interrupted: interrupted; no evidence was published"]
    assert_nothing_published(dirs, result)
    assert_gone(pid_file)


def test_re_inspect_write_failure_rolls_back(dirs):
    fake_path, pid_file = fake_r2(dirs, GOOD)
    def limit_file_size():
        resource.setrlimit(resource.RLIMIT_FSIZE, (64, 64))
    result = run_cli(
        dirs, "re", "inspect", dirs.sample, "--out", dirs.outputs / "evidence",
        path=fake_path, preexec_fn=limit_file_size,
    )
    assert result.returncode == 1
    [message] = messages(result.stderr)
    assert message.startswith("r2spec: failed: File too large")
    assert_nothing_published(dirs, result)
    assert_gone(pid_file)


@pytest.mark.parametrize("unbuffered", [False, True], ids=["buffered", "unbuffered"])
def test_re_inspect_closed_stdout_after_commit_still_exits_0(dirs, unbuffered):
    fake_path, pid_file = fake_r2(dirs, GOOD)
    out = dirs.outputs / "evidence"
    command, env = invocation(dirs, ["re", "inspect", dirs.sample, "--out", out], fake_path)
    if unbuffered:
        env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command, env=env, cwd=dirs.cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    process.stdout.close()  # The reader is gone before the CLI prints the evidence path.
    try:
        _, stderr = process.communicate(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode == 0, stderr
    assert stderr == ""
    assert published_path(out).is_file()
    assert_gone(pid_file)


# In-process OS-boundary injection: exit mapping and rollback only.
def test_re_inspect_link_failure_rolls_back_in_process(dirs, monkeypatch, capsys):
    fake_path, pid_file = fake_r2(dirs, GOOD)
    monkeypatch.setenv("PATH", fake_path)
    def refused(*args, **kwargs):
        raise OSError(errno.EIO, "injected link failure")
    monkeypatch.setattr(os, "link", refused)
    handler = signal.getsignal(signal.SIGINT)
    out = dirs.outputs / "evidence"
    assert cli.main(["re", "inspect", str(dirs.sample), "--out", str(out)]) == 1
    captured = capsys.readouterr()
    assert messages(captured.err) == ["r2spec: failed: injected link failure; no evidence was published"]
    assert captured.out == ""
    assert_nothing_published(dirs)
    assert signal.getsignal(signal.SIGINT) is handler
    assert_gone(pid_file)


def test_re_inspect_cleanup_pending_exits_5_in_process(dirs, monkeypatch, capsys):
    fake_path, pid_file = fake_r2(dirs, GOOD)
    monkeypatch.setenv("PATH", fake_path)
    def denied(pid, sig):
        raise PermissionError(errno.EPERM, "injected group-signal refusal")
    monkeypatch.setattr(_child, "_signal_group", denied)
    out = dirs.outputs / "evidence"
    try:
        assert cli.main(["re", "inspect", str(dirs.sample), "--out", str(out)]) == 5
        pid = pid_file.read_text()
        [message] = messages(capsys.readouterr().err)
        assert message.startswith(f"r2spec: cleanup_pending: radare2 child {pid} cleanup pending:")
        assert message.endswith("no evidence was published")
        assert_nothing_published(dirs)
    finally:
        if pid_file.exists():
            pid = int(pid_file.read_text())
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass


# Ledger: owner-only commands over the stdlib-only ledger with the shared exit statuses.
def ledger_lifecycle(dirs):
    target_sha = hashlib.sha256(dirs.sample.read_bytes()).hexdigest()
    evidence = dirs.targets / "evidence.json"
    evidence.write_text(json.dumps({"schema": "r2spec.re.inspect/1", "target": {"sha256": target_sha}}))
    test_file = dirs.targets / "test_sum.py"
    test_file.write_text("def test_sum():\n    pass\n")
    return [
        ["init"],
        ["change", "target", target_sha, "--reason", "inspected"],
        ["change", "configuration", "flake-1", "--reason", "pinned"],
        ["change", "specification", "spec-1", "--reason", "drafted"],
        ["change", "tests", "tests-1", "--reason", "written"],
        ["add", "sum-1", "--requirement", "sums the records"],
        ["evidence", "sum-1", "E1", evidence, "--kind", "re-inspect"],
        ["unknown", "sum-1", "overflow", "--reason", "not observable statically"],
        ["specify", "sum-1"],
        ["cover", "sum-1", "--test", test_file],
        ["result", "sum-1", "--outcome", "pass", "--evidence", "E1", "--target", target_sha,
         "--configuration", "flake-1", "--specification", "spec-1", "--tests", "tests-1"],
        ["close", "sum-1", "--accept", "--rationale", "passes at the current inputs", "--no-known-limitations"],
    ]


def run_ledger(dirs, path, *args):
    return run_cli(dirs, "ledger", *args, "--ledger", dirs.outputs / "ledger", path=path)


def test_ledger_cli_runs_the_obligation_lifecycle(dirs):
    path = empty_path(dirs)
    for revision, step in enumerate(ledger_lifecycle(dirs)):
        expect = [] if step[0] == "init" else ["--expect-revision", revision - 1]
        result = run_ledger(dirs, path, *step, *expect)
        assert (result.returncode, result.stdout, result.stderr) == (0, f"{revision}\n", "")
    shown = run_ledger(dirs, path, "show", "sum-1")
    assert (shown.returncode, shown.stderr) == (0, "")
    obligation = json.loads(shown.stdout)
    assert (obligation["state"], obligation["closure"]["limitations"]) == ("closed", [])
    assert obligation["unknowns"] == [{"field": "overflow", "reason": "not observable statically"}]
    assert json.loads(run_ledger(dirs, path, "show").stdout)["revision"] == 11
    ledger_dir = dirs.outputs / "ledger"
    assert stat.S_IMODE(ledger_dir.stat().st_mode) == 0o700
    assert "### sum-1: closed" in (ledger_dir / "LEDGER.md").read_text()
    (ledger_dir / "LEDGER.md").write_text("edited by hand")
    assert run_ledger(dirs, path, "render").stdout == "11\n"
    assert "### sum-1: closed" in (ledger_dir / "LEDGER.md").read_text()


def test_ledger_cli_maps_refusals_and_failures_to_exit_statuses(dirs):
    path = empty_path(dirs)
    ledger_dir = dirs.outputs / "ledger"
    assert run_ledger(dirs, path, "init").returncode == 0
    for args, status, message in [
        (["init"], 2, "invocation: the ledger directory already exists; name a new directory"),
        (["add", "../a", "--requirement", "r", "--expect-revision", "0"], 2,
         "invocation: obligation id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}"),
        (["evidence", "a-1", "E1", dirs.sample, "--kind", "guess", "--expect-revision", "0"], 2,
         "invocation: evidence kind must be one of re-inspect, test-result, other"),
        (["add", "a-1", "--requirement", "r", "--expect-revision", "5"], 3,
         "refused: stale ledger revision: expected 5, found 0"),
        (["specify", "a-9", "--expect-revision", "0"], 3, "refused: unknown obligation a-9"),
        (["show", "a-9"], 3, "refused: unknown obligation a-9"),
    ]:
        before = snapshot(dirs.outputs)
        result = run_ledger(dirs, path, *args)
        assert (result.returncode, result.stdout, messages(result.stderr)) == (status, "", [f"r2spec: {message}"])
        assert snapshot(dirs.outputs) == before
    (ledger_dir / "ledger.json").write_text("{not json")
    result = run_ledger(dirs, path, "render")
    assert (result.returncode, result.stdout, messages(result.stderr)) == (
        1, "", ["r2spec: failed: corrupt ledger: ledger.json is not valid JSON"],
    )


def test_ledger_cli_never_prints_raw_control_characters(dirs):
    path = empty_path(dirs)
    hostile = "\x1b]0;owned\x07\x1b[2J\x7f"
    assert run_ledger(dirs, path, "init").returncode == 0
    assert run_ledger(dirs, path, "add", "a-1", "--requirement", hostile, "--expect-revision", "0").returncode == 0
    shown = run_ledger(dirs, path, "show", "a-1")
    assert shown.returncode == 0
    assert json.loads(shown.stdout)["requirement"] == hostile
    assert not set(shown.stdout) & set("\x1b\x07\x7f")
    assert run_ledger(dirs, path, "change", "tests", hostile, "--reason", "r", "--expect-revision", "1").returncode == 0
    refused = run_ledger(dirs, path, "change", "tests", hostile, "--reason", "r", "--expect-revision", "2")
    assert refused.returncode == 3
    assert not set(refused.stderr) & set("\x1b\x07\x7f")
    assert messages(refused.stderr) == [r"r2spec: refused: tests is already \x1b]0;owned\x07\x1b[2J\x7f"]


def test_ledger_route_never_loads_the_re_stack(dirs):
    fake_path, pid_file = fake_r2(dirs, GOOD)
    result = run_cli(dirs, "ledger", "init", "--ledger", dirs.outputs / "ledger", path=fake_path, importtime=True)
    assert result.returncode == 0, result.stderr
    assert "r2spec.ledger" in imported(result.stderr)
    assert not imported(result.stderr) & RE_STACK
    assert not pid_file.exists()


def test_ledger_cli_closed_stdout_after_commit_still_exits_0(dirs):
    path = empty_path(dirs)
    ledger_dir = dirs.outputs / "ledger"
    assert run_ledger(dirs, path, "init").returncode == 0
    command, env = invocation(
        dirs, ["ledger", "add", "a-1", "--requirement", "r", "--ledger", ledger_dir, "--expect-revision", "0"], path,
    )
    process = subprocess.Popen(
        command, env=env, cwd=dirs.cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    process.stdout.close()  # The reader is gone before the CLI prints the new revision.
    try:
        _, stderr = process.communicate(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode == 0, stderr
    assert stderr == ""
    assert json.loads((ledger_dir / "ledger.json").read_text())["revision"] == 1


def test_ledger_cli_interrupt_after_commit_exits_130_naming_the_revision_in_process(dirs, monkeypatch, capsys):
    ledger_dir = dirs.outputs / "ledger"
    assert cli.main(["ledger", "init", "--ledger", str(ledger_dir)]) == 0
    capsys.readouterr()
    def delivered(signum, frame):
        raise AssertionError("SIGINT was delivered during the commit instead of being held")
    real = os.fsync
    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
        return real(fd)
    previous = signal.signal(signal.SIGINT, delivered)
    monkeypatch.setattr(os, "fsync", fsync)
    try:
        status = cli.main([
            "ledger", "add", "a-1", "--requirement", "r", "--ledger", str(ledger_dir), "--expect-revision", "0",
        ])
    finally:
        monkeypatch.setattr(os, "fsync", real)
        signal.signal(signal.SIGINT, previous)
    captured = capsys.readouterr()
    assert (status, captured.out) == (130, "")
    assert messages(captured.err) == ["r2spec: interrupted: interrupted; committed revision 1"]
    assert json.loads((ledger_dir / "ledger.json").read_text())["revision"] == 1
