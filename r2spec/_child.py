"""Bounded cleanup for an exclusively owned Linux child/process group.

The embedding process must not reap these children or change SIGCHLD semantics.
An unreaped leader pins the numeric group until the final signal. Pending handles
retain that leader for explicit retry; abandoning a handle abandons cleanup.
This is not containment of descendants that escape the process group.
"""
import os
import signal
import threading
import time


def _signal_group(pid: int, sig: int) -> None:
    os.killpg(pid, sig)


class ChildCleanup:
    """A live local cleanup handle, not a serializable or transferable capability.

    Callers must serialize retries with session cleanup. No background work,
    finalizer or interpreter-private reaper owns this handle.
    """

    def __init__(self, process=None):
        self._process = process
        self._state = "running"
        self._detail = ""
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        return self._state

    @property
    def detail(self) -> str:
        return self._detail

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    def observe(self) -> str:
        """Observe exit without reaping; other wait failures remain OS errors."""
        if self._state == "complete":
            return "exited"
        if self._state == "lost":
            return "lost"
        try:
            if self._process.returncode is not None:
                raise ChildProcessError("child was reaped outside its cleanup owner")
            result = os.waitid(
                os.P_PID, self._process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT,
            )
        except ChildProcessError:
            self._state = "lost"
            self._detail = "child ownership lost; further group signaling is prohibited"
            return "lost"
        return "exited" if result is not None else "running"

    def _wait_until(self, deadline: float) -> str:
        while True:
            status = self.observe()
            if status != "running" or time.monotonic() >= deadline:
                return status
            time.sleep(min(0.01, max(0, deadline - time.monotonic())))

    def finish(self) -> str:
        """Try TERM/grace, final KILL and reap within one 1.25-second budget."""
        return self._finish(grace=True)

    def retry(self) -> str:
        """Retry only cleanup: KILL and reap within one second, never execution."""
        return self._finish(grace=False)

    def _finish(self, *, grace: bool) -> str:
        if not self._lock.acquire(blocking=False):
            return self._state
        try:
            if self._state in ("complete", "lost"):
                return self._state
            if self._process is None:
                # No library-owned handle was acquired; Popen internals are separate.
                self._state = "complete"
                return self._state
            self._state = "pending"
            self._detail = ""
            deadline = time.monotonic() + (1.25 if grace else 1.0)
            try:
                if self.observe() == "lost":
                    return self._state
                if grace:
                    try:
                        _signal_group(self._process.pid, signal.SIGTERM)
                    except OSError:
                        # Only the final KILL determines signaling completion.
                        pass
                    if self._wait_until(min(deadline, time.monotonic() + 0.25)) == "lost":
                        return self._state
                try:
                    _signal_group(self._process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                if self._wait_until(deadline) != "exited":
                    if self._state != "lost":
                        self._detail = "child exit not observed before cleanup deadline"
                    return self._state
                # waitpid exposes ECHILD, unlike Popen.poll's synthesized zero.
                pid, status = os.waitpid(self._process.pid, os.WNOHANG)
                if pid == 0:
                    self._detail = "child could not be reaped before cleanup deadline"
                    return self._state
                self._process.returncode = os.waitstatus_to_exitcode(status)
                self._state = "complete"
            except ChildProcessError:
                self._state = "lost"
                self._detail = "child ownership lost; further group signaling is prohibited"
            except OSError as exc:
                self._detail = f"cleanup OS failure: {exc}"
            return self._state
        finally:
            self._lock.release()


def attach_cleanup(error: BaseException, owner: ChildCleanup | None) -> None:
    """Preserve the primary error while exposing incomplete, caller-owned cleanup.

    Attribute-rejecting foreign exceptions are outside the ownership-transfer
    contract. Notes alone are disclosure, not ownership.
    """
    if owner is None:
        return
    try:
        error.cleanup = owner
    except Exception:
        pass
    try:
        error.add_note(f"cleanup {owner.state}: {owner.detail}; retain cleanup and call retry()")
    except Exception:
        pass
