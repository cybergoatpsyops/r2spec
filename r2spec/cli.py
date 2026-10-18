"""Explicit, independent r2spec modes for ``python -m r2spec``.

This module imports only the standard library. The RE inspect handler imports
the RE stack lazily, so help and the clean-room refusal never load the
transport, start radare2 or touch the filesystem. The ledger handler imports
only the stdlib-only owner ledger, so ledger work never loads the RE stack.
"""
import argparse
import json
import os
import sys

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_REFUSED = 3
EXIT_UNAVAILABLE = 4
EXIT_CLEANUP_PENDING = 5
EXIT_INTERRUPTED = 130

_EXITS = {
    "failed": EXIT_FAILED,
    "invocation": EXIT_USAGE,
    "refused": EXIT_REFUSED,
    "unavailable": EXIT_UNAVAILABLE,
    "cleanup_pending": EXIT_CLEANUP_PENDING,
    "interrupted": EXIT_INTERRUPTED,
}
_REFUSAL = (
    "clean-room mode is not qualified in this build; "
    "no specification was read and no artifact was written."
)
_LEDGER_INPUTS = ("target", "configuration", "specification", "tests")


def _printable(text: str) -> str:
    """Escape non-printable characters so r2 or target text cannot drive the terminal."""
    return "".join(
        char if " " <= char <= "~" else char.encode("unicode_escape").decode("ascii")
        for char in text
    )


def _report(kind: str, message: str) -> None:
    print(f"r2spec: {kind}: {_printable(message)}", file=sys.stderr)


def _emit(text: str) -> None:
    """Print the result of an operation whose effect is already committed."""
    try:
        print(text, flush=True)
    except OSError:
        # The result is already committed; a closed stdout must not turn success
        # into failure, including the interpreter's final flush at shutdown.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        os.close(devnull)


def _re_inspect(args: argparse.Namespace) -> int:
    from r2spec import re_inspect  # RE-only runtime edge; other routes stay r2-free.

    try:
        evidence = re_inspect.run(args.target, args.out)
    except re_inspect.InspectError as error:
        _report(error.kind, str(error))
        return _EXITS[error.kind]
    _emit(_printable(evidence))
    return EXIT_OK


def _clean_room(args: argparse.Namespace) -> int:
    _report("refused", _REFUSAL)
    return EXIT_REFUSED


def _ledger(args: argparse.Namespace) -> int:
    from r2spec import ledger  # Stdlib-only owner ledger; never loads the RE stack.

    try:
        output = args.run(ledger, args)
    except ledger.LedgerError as error:
        _report(error.kind, str(error))
        return _EXITS[error.kind]
    _emit(str(output))
    return EXIT_OK


def _show(ledger, args: argparse.Namespace) -> str:
    document = ledger.load(args.ledger)[0]
    if args.id is not None:
        if args.id not in document["obligations"]:
            raise ledger.LedgerError("refused", f"unknown obligation {args.id}")
        document = document["obligations"][args.id]
    # ASCII JSON escapes every control character except DEL, which could still reach a terminal.
    return json.dumps(document, sort_keys=True, indent=2, ensure_ascii=True).replace("\x7f", "\\u007f")


def _add_ledger(modes) -> argparse.ArgumentParser:
    directory = argparse.ArgumentParser(add_help=False)
    directory.add_argument("--ledger", metavar="DIR", required=True, help="your ledger directory")
    revision = argparse.ArgumentParser(add_help=False)
    revision.add_argument(
        "--expect-revision", metavar="N", type=int, required=True,
        help="the ledger revision this change applies to; a stale revision is refused",
    )
    mode = modes.add_parser(
        "ledger", allow_abbrev=False, help="owner-controlled obligation ledger",
        description=(
            "Owner-only obligation ledger. DIR/ledger.json is the only authority and "
            "DIR/LEDGER.md is regenerated from it, never read. Each change prints the new "
            "revision. Covered means tests exist, not that they pass."
        ),
    )
    operations = mode.add_subparsers(dest="operation", title="operations", metavar="OPERATION")

    def operation(name, summary, run, *parents):
        parser = operations.add_parser(
            name, allow_abbrev=False, help=summary, description=summary, parents=[directory, *parents],
        )
        parser.set_defaults(handler=_ledger, run=run)
        return parser

    parser = operation(
        "init", "create a new private ledger directory at revision 0",
        lambda ledger, a: ledger.init(a.ledger, a.forbid_under or []),
    )
    parser.add_argument(
        "--forbid-under", metavar="PATH", action="append",
        help="a directory the ledger must never be inside, such as an Implementer workspace; repeatable",
    )
    parser = operation(
        "add", "add a pending obligation",
        lambda ledger, a: ledger.add(a.ledger, a.id, a.requirement, expect_revision=a.expect_revision),
        revision,
    )
    parser.add_argument("id", metavar="ID", help="new obligation id")
    parser.add_argument("--requirement", metavar="TEXT", required=True, help="semantic requirement")
    parser = operation(
        "specify", "mark a pending obligation specified under the declared specification",
        lambda ledger, a: ledger.specify(a.ledger, a.id, expect_revision=a.expect_revision),
        revision,
    )
    parser.add_argument("id", metavar="ID", help="obligation id")
    parser = operation(
        "evidence", "attach a private evidence file by path and sha256; its content is never copied",
        lambda ledger, a: ledger.attach_evidence(
            a.ledger, a.id, a.ref, a.path, kind=a.kind, expect_revision=a.expect_revision,
        ),
        revision,
    )
    parser.add_argument("id", metavar="ID", help="obligation id")
    parser.add_argument("ref", metavar="REF", help="new evidence reference")
    parser.add_argument("path", metavar="PATH", help="evidence file")
    parser.add_argument(
        "--kind", metavar="KIND", required=True, help="re-inspect, test-result or other",
    )
    parser = operation(
        "unknown", "record an explicit unknown about an obligation",
        lambda ledger, a: ledger.note_unknown(a.ledger, a.id, a.field, a.reason, expect_revision=a.expect_revision),
        revision,
    )
    parser.add_argument("id", metavar="ID", help="obligation id")
    parser.add_argument("field", metavar="FIELD", help="what is unknown")
    parser.add_argument("--reason", metavar="TEXT", required=True, help="why it is unknown")
    parser = operation(
        "cover", "record that tests exist for a specified obligation; this is not a pass",
        lambda ledger, a: ledger.cover(a.ledger, a.id, a.tests, expect_revision=a.expect_revision),
        revision,
    )
    parser.add_argument("id", metavar="ID", help="obligation id")
    parser.add_argument(
        "--test", dest="tests", metavar="PATH", action="append", required=True, help="test file; repeatable",
    )
    parser = operation(
        "result", "record a reported outcome at the inputs it ran against",
        lambda ledger, a: ledger.record_result(
            a.ledger, a.id, outcome=a.outcome, evidence=a.evidence,
            inputs={name: getattr(a, name) for name in _LEDGER_INPUTS}, expect_revision=a.expect_revision,
        ),
        revision,
    )
    parser.add_argument("id", metavar="ID", help="obligation id")
    parser.add_argument("--outcome", metavar="OUTCOME", required=True, help="pass, fail or error")
    parser.add_argument("--evidence", metavar="REF", required=True, help="attached evidence reference")
    for name in _LEDGER_INPUTS:
        parser.add_argument(f"--{name}", metavar="ID", required=True, help=f"{name} id the result ran against")
    parser = operation(
        "close", "close a covered obligation on your explicit acceptance of its latest passing result",
        lambda ledger, a: ledger.close(
            a.ledger, a.id, expect_revision=a.expect_revision, disposition="accepted",
            rationale=a.rationale, limitations=a.limitation or [],
        ),
        revision,
    )
    parser.add_argument("id", metavar="ID", help="obligation id")
    parser.add_argument("--accept", action="store_true", required=True, help="explicitly accept the closure")
    parser.add_argument("--rationale", metavar="TEXT", required=True, help="why the obligation is closed")
    limits = parser.add_mutually_exclusive_group(required=True)
    limits.add_argument("--limitation", metavar="TEXT", action="append", help="known limitation; repeatable")
    limits.add_argument("--no-known-limitations", action="store_true", help="declare that there are none")
    parser = operation(
        "change", "declare a changed input and reopen affected obligations",
        lambda ledger, a: ledger.change(
            a.ledger, a.dimension, a.input_id, reason=a.reason, affects=a.affects,
            expect_revision=a.expect_revision,
        ),
        revision,
    )
    parser.add_argument(
        "dimension", metavar="DIMENSION", help="target, configuration, specification, tests or unknown",
    )
    parser.add_argument("input_id", metavar="ID", help="new input id; the target id is its sha256")
    parser.add_argument("--reason", metavar="TEXT", required=True, help="why it changed")
    parser.add_argument(
        "--affects", metavar="ID", action="append", help="obligation to reopen; repeatable, default all",
    )
    parser = operation(
        "reopen", "move one obligation to a lower state",
        lambda ledger, a: ledger.reopen(a.ledger, a.id, to=a.to, reason=a.reason, expect_revision=a.expect_revision),
        revision,
    )
    parser.add_argument("id", metavar="ID", help="obligation id")
    parser.add_argument("--to", metavar="STATE", required=True, help="pending, specified or covered")
    parser.add_argument("--reason", metavar="TEXT", required=True, help="why it is reopened")
    operation("render", "regenerate LEDGER.md from the committed ledger", lambda ledger, a: ledger.render(a.ledger))
    parser = operation("show", "print the committed ledger, or one obligation, as JSON", _show)
    parser.add_argument("id", metavar="ID", nargs="?", help="obligation id")
    return mode


def _parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    parser = argparse.ArgumentParser(
        prog="python -m r2spec", allow_abbrev=False,
        description="Run one explicit, independent mode. Without a mode, print this help and exit 2.",
    )
    modes = parser.add_subparsers(dest="mode", title="modes", metavar="MODE")
    reverse = modes.add_parser(
        "re", allow_abbrev=False, help="reverse engineering of an authorized target",
        description="Reverse Engineering: static inspection of an authorized target.",
    )
    operations = reverse.add_subparsers(dest="operation", title="operations", metavar="OPERATION")
    inspect = operations.add_parser(
        "inspect", allow_abbrev=False, help="record private evidence with confidence and unknowns",
        description=(
            "Statically inspect TARGET with radare2 and publish private evidence to a new "
            "directory. Running this asserts your authorization to analyze TARGET. TARGET "
            "is never executed, but radare2 itself is not sandboxed."
        ),
    )
    inspect.add_argument("target", metavar="TARGET", help="existing regular file to inspect")
    inspect.add_argument(
        "--out", metavar="DIR", required=True, help="new evidence directory to create; must not exist",
    )
    inspect.set_defaults(handler=_re_inspect)
    clean_room = modes.add_parser(
        "clean-room", allow_abbrev=False, help="refuses: clean-room implementation is not qualified",
        description=(
            "Clean-Room Implementation is not qualified in this build. "
            "This route always refuses and does nothing else."
        ),
    )
    clean_room.set_defaults(handler=_clean_room)
    return parser, {"re": reverse, "ledger": _add_ledger(modes)}


def main(argv: list[str] | None = None) -> int:
    try:
        parser, groups = _parser()
        args = parser.parse_args(argv)
        if args.mode is None:
            parser.print_help(sys.stderr)
            return EXIT_USAGE
        if args.mode in groups and args.operation is None:
            groups[args.mode].print_help(sys.stderr)
            return EXIT_USAGE
        return args.handler(args)
    except KeyboardInterrupt:
        _report("interrupted", "operation interrupted")
        return EXIT_INTERRUPTED
