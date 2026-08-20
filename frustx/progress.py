"""Terminal progress reporting for the CLI.

A real run is minutes to hours -- 1000 decoys is roughly 12 min at --protocol min
and about an hour at "relax" -- so the one thing the user needs from the terminal
is an answer to "how long is this going to take". The previous counter printed
`decoy 347/1000` and left them to do that arithmetic themselves.

Stdlib only, deliberately: tqdm and rich would each be a runtime dependency of a
package that otherwise needs only numpy/pandas/biopython, and the whole renderer
is thirty lines. `frustx.cli` already refuses click for the same reason.

Nothing here is imported by the compute path. `compute_frustration` takes a plain
callable(done, total); this module just supplies a nicer one than a print.
"""

import shutil
import sys
import time

_FILL, _EMPTY = "█", "░"          # full block, light shade
_FILL_ASCII, _EMPTY_ASCII = "#", "-"

# Redraw no more often than this. On 8 workers at --protocol min, decoys land
# several times a second; every one is a write+flush that may be going down an ssh
# connection, and the eye cannot read them anyway.
_MIN_REDRAW_INTERVAL = 0.1

# Without a terminal, "\r" is not a cursor movement -- it is a byte in the log
# file, and a 1000-decoy run becomes one unreadable line. Redirected output gets
# ordinary newline-terminated lines at this granularity instead.
_LOG_EVERY_PERCENT = 10


def format_duration(seconds):
    """Compact h/m/s. Truncates rather than rounds, so a countdown never shows a
    number larger than the time actually left."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def format_bar(done, total, elapsed, width=80, ascii_only=False):
    """Render one progress line. Pure, so the layout is testable without a
    terminal; `width` is the number of columns available.

    The ETA extrapolates from the mean time per completed decoy over the whole run
    so far, not from a recent window. Decoys are the same amount of work as one
    another, so the running mean is the better estimator here -- a windowed rate
    would swing on every scheduling hiccup and on the ragged start of a Pool.
    """
    frac = min(max(done / total, 0.0), 1.0) if total else 1.0
    # Floor, not round, so 999/1000 reads 99% and 100% means finished.
    pct = int(frac * 100)
    eta = format_duration(elapsed * (total - done) / done) if done > 0 else "--"
    prefix = "  decoys "
    tail = f" {pct:3d}%  {done}/{total}  {format_duration(elapsed)} elapsed, {eta} left"
    # The bar gets whatever the text leaves, capped so it does not sprawl across a
    # wide terminal.
    n = min(40, width - len(prefix) - len(tail) - 1)
    if n < 8:
        # No room for a bar worth looking at, so drop it and keep the numbers, which
        # are the part carrying the information. Truncated to fit because a line that
        # WRAPS breaks \r for good: the next redraw returns to the start of the
        # wrapped row and the first row stays on screen forever.
        return f"{prefix.rstrip()}{tail}"[:max(1, width - 1)]
    fill, empty = (_FILL_ASCII, _EMPTY_ASCII) if ascii_only else (_FILL, _EMPTY)
    filled = int(n * frac)
    return f"{prefix}{fill * filled}{empty * (n - filled)}{tail}"


class Reporter:
    """Callable matching compute_frustration's `progress` contract, (done, total).

    Owns the clock and the terminal so the CLI stays wiring. Construct it where the
    run starts: `started` is set here and is what the ETA counts from.
    """

    def __init__(self, stream=None, enabled=True, isatty=None):
        self.stream = sys.stderr if stream is None else stream
        self.enabled = enabled
        self.started = time.time()
        self._last_draw = -_MIN_REDRAW_INTERVAL     # let the first call through
        self._last_bucket = -1
        self._line_len = 0
        # isatty is injectable only so the tests can exercise both branches against
        # a StringIO; nothing else should pass it.
        self.tty = (bool(getattr(self.stream, "isatty", lambda: False)())
                    if isatty is None else isatty)
        # A stderr that cannot encode the block characters would raise
        # UnicodeEncodeError out of the progress callback and take the whole run
        # down with it -- hours of decoys lost to a cosmetic glyph. Ask once, here.
        try:
            _FILL.encode(getattr(self.stream, "encoding", None) or "ascii")
            self.ascii_only = False
        except (UnicodeEncodeError, LookupError):
            self.ascii_only = True

    def __call__(self, done, total):
        if not self.enabled:
            return
        elapsed = time.time() - self.started
        if self.tty:
            # The final line is the one that stays on screen, so it is drawn
            # whatever the throttle says.
            if done < total and elapsed - self._last_draw < _MIN_REDRAW_INTERVAL:
                return
            self._last_draw = elapsed
            line = format_bar(done, total, elapsed,
                              shutil.get_terminal_size((80, 24)).columns,
                              self.ascii_only)
            # Pad to the previous line's length: "1h02m left" -> "58m30s left" is
            # shorter, and without this the tail of the old line survives on screen.
            pad = " " * max(0, self._line_len - len(line))
            self._line_len = len(line)
            self.stream.write("\r" + line + pad)
        else:
            pct = int(done / total * 100) if total else 100
            bucket = pct // _LOG_EVERY_PERCENT
            if done < total and bucket == self._last_bucket:
                return
            self._last_bucket = bucket
            eta = format_duration(elapsed * (total - done) / done) if done > 0 else "--"
            self.stream.write(f"  decoys {done}/{total} ({pct}%)  "
                              f"{format_duration(elapsed)} elapsed, {eta} left\n")
        self.stream.flush()

    def finish(self):
        """Close the line so whatever prints next starts at column 0. A no-op when
        nothing was ever drawn, which is what keeps --quiet from emitting a stray
        blank line."""
        if self.enabled and self.tty and self._line_len:
            self.stream.write("\n")
            self.stream.flush()
