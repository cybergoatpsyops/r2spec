"""Explicit, independent r2spec modes for ``python -m r2spec``.

This module imports only argparse and sys. The RE inspect handler imports the
RE stack lazily, so help and the clean-room refusal never load the transport,
start radare2 or touch the filesystem.
"""
import argparse
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
    "unavailable": EXIT_UNAVAILABLE,
    "cleanup_pending": EXIT_CLEANUP_PENDING,
    "interrupted": EXIT_INTERRUPTED,
}
_REFUSAL = (
    "clean-room mode is not qualified in this build; "
    "no specification was read and no artifact was written."
)


def _printable(text: str) -> str:
    """Escape non-printable characters so r2 or target text cannot drive the terminal."""
    return "".join(
        char if " " <= char <= "~" else char.encode("unicode_escape").decode("ascii")
        for char in text
    )


def _report(kind: str, message: str) -> None:
    print(f"r2spec: {kind}: {_printable(message)}", file=sys.stderr)


def _re_inspect(args: argparse.Namespace) -> int:
    from r2spec import re_inspect  # RE-only runtime edge; other routes stay r2-free.

    try:
        evidence = re_inspect.run(args.target, args.out)
    except re_inspect.InspectError as error:
        _report(error.kind, str(error))
        return _EXITS[error.kind]
    print(_printable(evidence))
    return EXIT_OK


def _clean_room(args: argparse.Namespace) -> int:
    _report("refused", _REFUSAL)
    return EXIT_REFUSED


def _parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
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
    return parser, reverse


def main(argv: list[str] | None = None) -> int:
    try:
        parser, reverse = _parser()
        args = parser.parse_args(argv)
        if args.mode is None:
            parser.print_help(sys.stderr)
            return EXIT_USAGE
        if args.mode == "re" and args.operation is None:
            reverse.print_help(sys.stderr)
            return EXIT_USAGE
        return args.handler(args)
    except KeyboardInterrupt:
        _report("interrupted", "operation interrupted")
        return EXIT_INTERRUPTED
