"""
Inject the population-point colouring overlay (code/point_coloring_overlay.js)
into an already-built scenario-explorer HTML, without a full rebuild.

The overlay is pure client-side JS that reads globals the explorer already
embeds, so this just splices the snippet in before </body>. Idempotent: a file
that already contains the overlay is left unchanged.

Usage:
    python3 code/add_point_coloring.py INPUT.html [OUTPUT.html]
If OUTPUT is omitted the input is edited in place (a .bak copy is kept).
"""
import shutil, sys
from pathlib import Path

MARK = "pt-color-panel"  # sentinel proving the overlay is already present
SNIPPET = Path(__file__).resolve().parent / "point_coloring_overlay.js"


def inject(html: str, snippet: str) -> str:
    if MARK in html:
        return html  # already injected
    if "</body>" in html:
        return html.replace("</body>", snippet + "\n</body>", 1)
    return html + snippet  # no body tag (defensive)


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    src = Path(argv[0])
    dst = Path(argv[1]) if len(argv) > 1 else src
    snippet = SNIPPET.read_text()
    html = src.read_text()
    if MARK in html:
        print(f"already injected, no change: {src}")
        return 0
    out = inject(html, snippet)
    if dst == src:
        shutil.copyfile(src, src.with_suffix(src.suffix + ".bak"))
    dst.write_text(out)
    print(f"injected overlay -> {dst}  (+{len(out)-len(html)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
