"""Tests for the CLI progress reporter.

Pure formatting and stream behaviour -- no PyRosetta, no terminal. Every expected
value here is checkable by mental arithmetic: 250 decoys done in 15 min is 3.6 s
each, so the remaining 750 are 45 min.
"""

import io
import time

from frustx.progress import Reporter, format_bar, format_duration


class _Stream:
    """The three things Reporter asks of a stream: write, flush, encoding.

    Not a StringIO subclass -- `encoding` is a read-only property there. A real
    sys.stderr reports "utf-8"; the parameter exists so the ASCII fallback can be
    exercised too.
    """

    def __init__(self, encoding="utf-8"):
        self.encoding = encoding
        self._buf = io.StringIO()

    def write(self, s):
        return self._buf.write(s)

    def flush(self):
        pass

    def getvalue(self):
        return self._buf.getvalue()


def test_format_duration_at_the_unit_boundaries():
    assert format_duration(0) == "0s"
    assert format_duration(59) == "59s"
    assert format_duration(60) == "1m00s"
    assert format_duration(3599) == "59m59s"
    assert format_duration(3600) == "1h00m"
    assert format_duration(45000) == "12h30m"
    # A negative estimate can only come from clock skew; it must not print "-1s".
    assert format_duration(-5) == "0s"


def test_eta_extrapolates_the_mean_rate():
    # 250 of 1000 in 900 s -> 3.6 s each -> 750 * 3.6 = 2700 s = 45m00s.
    line = format_bar(250, 1000, 900.0, width=80)
    assert "15m00s elapsed, 45m00s left" in line
    assert " 25%" in line


def test_no_eta_before_the_first_decoy_lands():
    assert "-- left" in format_bar(0, 1000, 0.0, width=80)


def test_percent_floors_so_that_100_means_finished():
    assert " 99%" in format_bar(999, 1000, 3596.0, width=80)
    assert "100%" in format_bar(1000, 1000, 3600.0, width=80)


def test_the_bar_fills_only_when_the_run_is_over():
    """A bar that reads full at 999/1000 is a lie the user acts on."""
    assert "░" in format_bar(999, 1000, 3596.0, width=80)
    assert "░" not in format_bar(1000, 1000, 3600.0, width=80)


def test_the_line_always_fits_the_terminal():
    """Overflow is not cosmetic: a wrapped line breaks \\r for the rest of the run,
    because the next redraw returns to the start of the wrapped row."""
    for width in (20, 40, 60, 62, 80, 120, 200):
        for done in (0, 1, 500, 999, 1000):
            line = format_bar(done, 1000, done * 3.6, width=width)
            assert len(line) < width, (width, done, len(line), line)


def test_a_narrow_terminal_keeps_the_numbers_and_drops_the_bar():
    line = format_bar(500, 1000, 1800.0, width=60)
    assert "500/1000" in line
    assert "░" not in line and "█" not in line


def test_ascii_fallback_when_the_stream_cannot_encode_blocks():
    assert Reporter(stream=_Stream("utf-8"), isatty=True).ascii_only is False
    assert Reporter(stream=_Stream("ascii"), isatty=True).ascii_only is True
    # A stream that cannot encode a block would otherwise raise UnicodeEncodeError
    # out of the progress callback and abort an hour-long run over a glyph.
    out = _Stream("ascii")
    Reporter(stream=out, isatty=True)(500, 1000)
    assert "#" in out.getvalue()


def test_on_a_tty_it_rewrites_one_line():
    out = _Stream()
    r = Reporter(stream=out, isatty=True)
    r(1, 1000)
    r.started -= 10          # get past the redraw throttle
    r(2, 1000)
    text = out.getvalue()
    assert text.count("\r") == 2
    assert "\n" not in text   # nothing scrolls until finish()
    r.finish()
    assert out.getvalue().endswith("\n")


def test_the_redraw_throttle_never_swallows_the_last_decoy():
    """Consecutive decoys inside the throttle window are dropped, but the final
    line is the one left on screen, so it must always be drawn."""
    out = _Stream()
    r = Reporter(stream=out, isatty=True)
    for k in range(1, 1000):
        r(k, 1000)
    assert out.getvalue().count("\r") == 1     # only the first got through
    r(1000, 1000)
    assert out.getvalue().count("\r") == 2
    assert "1000/1000" in out.getvalue()


def test_a_shrinking_line_is_padded_over():
    """"1h02m left" -> "58m30s left" is shorter; without padding the tail of the
    old line survives on screen."""
    out = _Stream()
    r = Reporter(stream=out, isatty=True)
    r._line_len = 120
    r(500, 1000)
    assert len(out.getvalue()) - 1 == 120      # minus the leading \r


def test_redirected_output_is_plain_lines_not_carriage_returns():
    """A log file full of \\r is one unreadable line."""
    out = _Stream()
    r = Reporter(stream=out, isatty=False)
    for k in range(1, 1001):
        r(k, 1000)
    r.finish()
    text = out.getvalue()
    assert "\r" not in text
    # First decoy, then every 10% boundary: 1, 100, 200, ... 1000.
    assert len(text.splitlines()) == 11
    assert text.splitlines()[-1].startswith("  decoys 1000/1000 (100%)")


def test_quiet_writes_nothing_at_all():
    out = _Stream()
    r = Reporter(stream=out, isatty=True, enabled=False)
    r(500, 1000)
    r.finish()
    assert out.getvalue() == ""


def test_reporter_matches_the_progress_callback_contract():
    """compute_frustration calls progress(done, total) positionally."""
    out = _Stream()
    r = Reporter(stream=out, isatty=True)
    assert r(7, 100) is None
    assert isinstance(r.started, float) and r.started <= time.time()
