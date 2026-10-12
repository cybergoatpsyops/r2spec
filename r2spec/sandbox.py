"""Nix runtime-closure executor containment, not coding-agent clean-room proof.

Runtime roots and spec/work directories are trusted deployment configuration, not
model-selected tool arguments. The caller owns semantic transfer approval. This
module neither qualifies agent history nor prevents an unrestricted caller from
reading originals outside this executor.
"""
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import subprocess
import threading
import time
from collections.abc import Sequence

from r2spec._child import ChildCleanup, attach_cleanup

_STORE = Path("/nix/store")
_HOST_TREES = tuple(map(Path, ("/nix", "/proc", "/sys", "/dev", "/run")))
_STORE_NAME = re.compile(r"[0-9abcdfghijklmnpqrsvwxyz]{32}-[A-Za-z0-9+._?=-]+")


class SandboxError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.cleanup: ChildCleanup | None = None


def _store_path(value: str | os.PathLike[str]) -> Path:
    try:
        path = Path(value)
        if (
            path.parent != _STORE or not _STORE_NAME.fullmatch(path.name)
            or not path.is_dir() or path.resolve(strict=True) != path
        ):
            raise ValueError("not a canonical store directory")
        return path
    except (OSError, TypeError, ValueError) as exc:
        raise SandboxError("input", "runtime roots must be exact existing Nix store directories") from exc


def _directory(value: str | os.PathLike[str]) -> Path:
    try:
        path = Path(value).absolute()
        if (
            not path.is_dir() or path.resolve(strict=True) != path
            or path == Path("/") or any(path.is_relative_to(root) for root in _HOST_TREES)
        ):
            raise ValueError("invalid mount directory")
        return path
    except (OSError, TypeError, ValueError) as exc:
        raise SandboxError("input", "mounts must be canonical existing directories outside host runtime and kernel trees") from exc


class Sandbox:
    """Execute approved runtimes with readonly specs and one writable workspace.

    Only local store queries occur; this class never fetches or installs packages.
    Rejecting known r2 binaries catches the development-closure mistake, but is
    not an audit of arbitrary caller-supplied runtime contents.
    """

    def __init__(
        self,
        runtime_roots: Sequence[str | os.PathLike[str]],
        *,
        executable: str | os.PathLike[str] = "bwrap",
    ):
        if (
            not isinstance(runtime_roots, Sequence)
            or isinstance(runtime_roots, (str, bytes, bytearray, os.PathLike))
            or not runtime_roots
        ):
            raise SandboxError("input", "provide an explicit nonempty runtime root list")
        roots = tuple(_store_path(root) for root in runtime_roots)
        self._program = shutil.which(os.fspath(executable))
        nix_store = shutil.which("nix-store")
        if self._program is None or nix_store is None:
            raise SandboxError("unavailable", "bubblewrap and local nix-store queries are required")
        self._program = str(Path(self._program).resolve(strict=True))
        try:
            query = subprocess.run(
                [nix_store, "--query", "--requisites", *map(str, roots)],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxError("unavailable", "cannot query the installed runtime closure") from exc
        if query.returncode != 0:
            raise SandboxError("input", "runtime closure is not available locally")
        paths = {_store_path(line) for line in query.stdout.splitlines() if line}
        if not set(roots).issubset(paths):
            raise SandboxError("input", "runtime query omitted a requested root")
        for path in paths:
            if any((path / "bin" / name).exists() for name in ("r2", "radare2")):
                raise SandboxError("input", "runtime closure contains radare2; the development shell is not allowed")
        self._paths = tuple(sorted(paths))
        self._path_env = ":".join(str(path / "bin") for path in self._paths if (path / "bin").is_dir())

    def run(
        self,
        command: Sequence[str],
        *,
        specs: str | os.PathLike[str],
        work: str | os.PathLike[str],
        timeout: float = 5,
        max_output_bytes: int = 65536,
        cancel: threading.Event | None = None,
    ) -> subprocess.CompletedProcess:
        """Run an approved runtime; nonzero exits are results, not successful builds.

        Incomplete cleanup raises a typed error or annotates the primary error.
        Retain exception.cleanup and explicitly retry it before discarding the
        handle. There is no background reaper; the host must not reap our child.
        """
        if (
            isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0
            or type(max_output_bytes) is not int or max_output_bytes <= 0
        ):
            raise SandboxError("input", "deadline and combined output limit must be positive and finite")
        specs_path, work_path = _directory(specs), _directory(work)
        if specs_path.is_relative_to(work_path) or work_path.is_relative_to(specs_path):
            raise SandboxError("input", "spec and work directories must not overlap")
        if (
            not isinstance(command, Sequence)
            or isinstance(command, (str, bytes, bytearray, os.PathLike)) or not command
            or any(not isinstance(arg, str) or "\0" in arg for arg in command)
        ):
            raise SandboxError("input", "command must be a nonempty argument list")
        try:
            program = Path(command[0])
            if not program.is_absolute():
                raise ValueError("use an absolute runtime executable")
            program = program.resolve(strict=True)
            if (
                not program.is_file() or not os.access(program, os.X_OK)
                or not any(program.is_relative_to(root) for root in self._paths)
            ):
                raise ValueError("executable outside approved runtime")
        except (OSError, ValueError) as exc:
            raise SandboxError("input", "executable must belong to the approved runtime closure") from exc

        if cancel is not None and cancel.is_set():
            raise SandboxError("cancelled", "execution cancelled before startup")
        arguments = [
            self._program, "--unshare-user", "--unshare-pid", "--unshare-net",
            "--unshare-ipc", "--unshare-uts", "--die-with-parent", "--new-session",
            "--cap-drop", "ALL", "--clearenv",
        ]
        for path in self._paths:
            arguments.extend(["--ro-bind", str(path), str(path)])
        arguments.extend([
            "--ro-bind", str(specs_path), "/specs", "--bind", str(work_path), "/work",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--chdir", "/work",
            "--setenv", "HOME", "/tmp", "--setenv", "TMPDIR", "/tmp",
            "--setenv", "PATH", self._path_env,
            "--", str(program), *command[1:],
        ])
        deadline = time.monotonic() + timeout
        child = ChildCleanup()
        primary = None
        selector = None
        output = {"stdout": bytearray(), "stderr": bytearray()}
        total = 0
        try:
            try:
                # The first store of a returned handle establishes ownership.
                child._process = subprocess.Popen(
                    arguments, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, env={}, close_fds=True, start_new_session=True,
                )
            except OSError as exc:
                raise SandboxError("unavailable", "cannot start bubblewrap") from exc
            process = child._process
            selector = selectors.DefaultSelector()
            for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            while True:
                status = child.observe()
                if status == "lost":
                    raise SandboxError("process", "executor child ownership lost")
                if not selector.get_map() and status == "exited":
                    break
                if cancel is not None and cancel.is_set():
                    raise SandboxError("cancelled", "execution cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SandboxError("timeout", "executor deadline exceeded")
                for key, _ in selector.select(min(remaining, 0.05)):
                    try:
                        chunk = os.read(key.fd, min(65536, max_output_bytes - total + 1))
                    except BlockingIOError:
                        continue
                    except OSError as exc:
                        raise SandboxError("process", "executor output pipe failed") from exc
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    total += len(chunk)
                    if total > max_output_bytes:
                        raise SandboxError("overflow", "combined executor output exceeds byte limit")
                    output[key.data].extend(chunk)
        except OSError as exc:
            primary = SandboxError("process", "executor observation or pipe failed")
            raise primary from exc
        except BaseException as exc:
            primary = exc
            raise
        finally:
            failure = None
            resources = [selector]
            if child._process is not None:
                resources.extend((child._process.stdout, child._process.stderr))
            for resource in resources:
                if resource is not None:
                    try:
                        resource.close()
                    except OSError as exc:
                        failure = SandboxError("cleanup", f"descriptor cleanup failed: {exc}")
            state = child.finish()
            owner = child if state != "complete" else None
            if owner is not None:
                failure = SandboxError("cleanup", child.detail)
                attach_cleanup(failure, owner)
            if primary is not None:
                attach_cleanup(primary, owner)
                if failure is not None and owner is None:
                    primary.add_note(str(failure))
            elif failure is not None:
                raise failure
        return subprocess.CompletedProcess(
            list(command), child.returncode,
            bytes(output["stdout"]), bytes(output["stderr"]),
        )
