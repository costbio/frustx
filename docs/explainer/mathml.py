"""Turn LaTeX in the template into MathML at build time.

Why MathML and not KaTeX/MathJax: the artifact CSP allows exactly one external
host (Google Fonts). A JS typesetter would have to be inlined in full, fonts and
all, for a page that has three equations. MathML is native in every current
browser, needs no runtime, and -- the actual reason we are here -- it does the
alignment itself. The previous version hand-set the equations as monospace text
with <sub> tags, and <sub> glyphs are narrower than a monospace cell, so a
hand-counted fraction bar could never stay in register.

Delimiters follow the LaTeX convention: \\[ ... \\] displays, \\( ... \\) is inline.
Conversion happens here, in the build, so the shipped page carries no dependency.
"""

import re

import latex2mathml.converter

# Non-greedy, DOTALL so a display equation may be written across several lines.
_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.S)
_INLINE = re.compile(r"\\\((.+?)\\\)", re.S)


# latex2mathml renders \underbrace as a literal U+23DF glyph and relies on the font
# to stretch it to the width of the group. That needs an OpenType MATH table with the
# glue parts for that character; without one -- and there is no guarantee the reader
# has one -- it stays a 19px stub under a 200px group, which looks broken. Dropping
# the glyph and drawing the rule in CSS is font-independent and stretches by
# construction, because a border is exactly as wide as the box it is on.
# Anchored on the INNERMOST munder: \underbrace nests two of them (brace, then
# label), and a plain non-greedy match would start at the outer one and leave it
# unclosed -- which the browser silently repairs by putting the label inline.
_UNDERBRACE = re.compile(r"<munder>((?:(?!<munder>).)+?)<mo>&#x23DF;</mo></munder>", re.S)


def _convert(latex: str, display: bool) -> str:
    tex = " ".join(latex.split())  # newlines in the source are layout, not math
    out = latex2mathml.converter.convert(tex)
    out = _UNDERBRACE.sub(r'<mrow class="ubrace">\1</mrow>', out)
    if display:
        # The converter always emits display="inline"; block equations need
        # display="block" so mfrac/msubsup get full-size (displaystyle) rendering.
        out = out.replace('display="inline"', 'display="block"', 1)
    cls = "m-block" if display else "m-inline"
    return out.replace("<math ", f'<math class="{cls}" ', 1)


def render(html: str) -> tuple[str, int]:
    """Replace every LaTeX span in `html`. Returns (html, count) so the caller can
    assert the template still contains the equations it thinks it does."""
    n = 0

    def sub(m, display):
        nonlocal n
        n += 1
        return _convert(m.group(1), display)

    html = _DISPLAY.sub(lambda m: sub(m, True), html)
    html = _INLINE.sub(lambda m: sub(m, False), html)
    return html, n
