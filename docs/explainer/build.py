"""Assemble the explainer page: git-tracked template + DVC-tracked numbers.

Split this way on purpose. The template carries no data, so it lives in git; the
payload is derived output, so it lives in results/ under DVC. Neither is committed
in the other's place. Run scripts/artefact_payload.py first if results/ has moved.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mathml import render as render_math  # noqa: E402  (needs the path above)

ROOT = Path(__file__).resolve().parent.parent.parent
TPL = ROOT / "docs/explainer/template.html"
PAYLOAD = ROOT / "results/artefact/payload.json"
OUT = ROOT / "results/artefact/explainer.html"

payload = json.loads(PAYLOAD.read_text())          # parsed to fail loudly on bad JSON
html = TPL.read_text()
assert "__PAYLOAD__" in html, "template lost its payload placeholder"

# LaTeX -> MathML before the payload goes in, so the math pass never sees the JSON
# (which is full of braces and backslashes it would have no business touching).
html, n_math = render_math(html)
assert n_math >= 20, f"only {n_math} equations converted -- did a delimiter get mangled?"
assert "\\(" not in html and "\\[" not in html, "an unclosed LaTeX delimiter survived"
# A leftover U+23DF means the underbrace rewrite missed a case and the reader would
# get an unstretched 19px stub instead of a rule.
assert "23DF" not in html, "an \\underbrace glyph escaped the CSS rewrite"
# The JSON sits inside a <script> block, so the only sequence that can break out of
# it is a literal "</script>"; escaping the slash keeps it inert and still valid JSON.
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")))
print(f"{OUT}  {OUT.stat().st_size/1024:.1f} KB  ({n_math} equations typeset)")
