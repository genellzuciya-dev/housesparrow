"""
Patch the ALREADY-GENERATED sparrow HTML maps with the new stacked left rail.

No refetching, no Earth Engine, no retraining. It reuses the feature-importance
numbers already baked into each HTML file, rips out the two old overlay blocks,
and injects the new rail in their place.

    python3 patch_maps.py sparrow_animation.html sparrow_seasonal.html

Writes <name>_patched.html next to each input. Originals are untouched.
Use this for layout-only changes; rerun sparrow_model.py when the MODEL changes.
"""
import json
import re
import sys

sys.path.insert(0, ".")


def _load_rail_builder(script="sparrow_model.py"):
    """Pull _left_rail_html (and its two dependencies) out of the edited script
    without importing the whole module, so nothing tries to reach Earth Engine."""
    src = open(script).read()
    ns = {}
    exec(re.search(r"SEASON_PALETTE = \{.*?\n\}\n", src, re.S).group(0), ns)
    exec(re.search(r"^DRIVERS = \[.*?\]\n", src, re.S | re.M).group(0), ns)
    exec(re.search(r"def _left_rail_html.*?(?=\ndef render_monthly_animation)",
                   src, re.S).group(0), ns)
    return ns["_left_rail_html"]


def _extract_importances(html):
    """Recover the season->feature->value dict from the embedded IMP payload."""
    i = html.index('var IMP = ')
    j = html.index("\n", i)
    blob = html[i + len('var IMP = '):j].rstrip().rstrip(";")
    return json.loads(blob)["data"]


def _strip_old_overlays(html):
    """Remove the standalone legend div and the whole impChart card + script."""
    # legend: the fixed div that starts with <b>Predicted suitability</b>
    pat = re.compile(
        r'<div style="position:fixed;top:150px;left:50px;.*?</div>\s*(?=<div)',
        re.S)
    html, n_legend = pat.subn("", html, count=1)

    # impChart card: from its opening div through its closing </div>
    i = html.index('<div id="impChart"')
    end = html.index("</script>", i) + len("</script>")
    # walk back to include the card's own closing tags before the <script>
    html = html[:i] + html[end:]
    return html, n_legend


def patch(path, rail_builder):
    html = open(path).read()
    importances = _extract_importances(html)

    if "animation" in path:
        note = ("Dark dots = sightings that month &nbsp;&middot;&nbsp; "
                "press play at the bottom of the map.")
        initial = "winter"
    else:
        note = "Check one season at a time (top-right) to compare."
        initial = next((s for s in ["winter", "spring", "summer", "fall"]
                        if s in importances), "spring")

    html, n_legend = _strip_old_overlays(html)
    rail = rail_builder(importances, note=note, initial_season=initial)
    html = html.replace("<body>", "<body>\n" + rail, 1)

    out = path.replace(".html", "_patched.html")
    open(out, "w").write(html)
    print(f"  {path}: removed {n_legend} legend + 1 chart, wrote {out}")


if __name__ == "__main__":
    build = _load_rail_builder()
    targets = sys.argv[1:] or ["sparrow_animation.html", "sparrow_seasonal.html"]
    print("Patching maps (layout only, model untouched):")
    for t in targets:
        patch(t, build)
    print("Done. Open the *_patched.html files.")
