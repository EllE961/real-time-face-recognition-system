"""Plot the LFW results written by scripts/evaluate_lfw.py.

    python scripts/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BG, PANEL = "#0d0c0b", "#131211"
INK, INK2, INK3 = "#ede7de", "#aaa299", "#7a736c"
GRID = "#24221f"
EMBER, TEMPER, PURPLE, GREEN = "#d95926", "#3987e5", "#9085e9", "#199e70"
SERIES = [EMBER, TEMPER, PURPLE, GREEN, "#e8b44c", "#c0509c"]
HEAT = LinearSegmentedColormap.from_list("heat", ["#141110", "#2a1410", "#4b170c", "#7a1f0a", "#ad320c",
                                                  "#d9520f", "#f07a1e", "#f9a63b", "#fccf6b", "#fff0c2"])
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.titlesize": 12.5, "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "axes.axisbelow": True, "font.family": ["Segoe UI", "DejaVu Sans"], "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK2,
    "figure.dpi": 110, "savefig.dpi": 200,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "assets")


def main():
    m = json.load(open(os.path.join(RES, "lfw_metrics.json")))
    d = np.load(os.path.join(RES, "lfw_distances.npz"))
    sw = m["sweep"]
    t = np.array(sw["tolerance"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    lines = [("correct", "enrolled person, named correctly", GREEN), ("unknown", "enrolled person, called Unknown", TEMPER),
             ("wrong", "enrolled person, named as someone else", EMBER),
             ("stranger_named", "stranger, named as an enrolled person", PURPLE)]
    i06 = int(np.argmin(np.abs(t - 0.6)))
    for key, label, color in lines:
        y = np.array(sw[key]) * 100
        a1.plot(t, y, color=color, label=label)
        if key in ("correct", "stranger_named"):
            a1.text(0.59, y[i06] + (-6 if key == "correct" else 2), f"{y[i06]:.1f}%", color=color, fontsize=9,
                    va="bottom", ha="right")
    a1.axvline(0.6, color=INK3, lw=1, ls=(0, (3, 3)))
    a1.text(0.605, 104, "app default 0.6", color=INK3, fontsize=8.5, ha="left", va="bottom")
    a1.set_xlim(t[0], t[-1])
    a1.set_ylim(0, 112)
    a1.set_xlabel("Tolerance setting")
    a1.set_ylabel("% of queries")
    a1.set_title("What the app answers, by tolerance")
    a1.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), fontsize=8.5, ncol=2)

    bins = np.linspace(0.1, 1.0, 46)
    for key, label, color in (("genuine", "same person (nearest enrolled photo)", GREEN),
                              ("other_person", "nearest other enrolled person", EMBER),
                              ("stranger", "stranger to nearest enrolled person", PURPLE)):
        x = d[key][np.isfinite(d[key])]
        a2.hist(x, bins=bins, density=True, histtype="stepfilled", alpha=0.28, color=color, edgecolor="none")
        a2.hist(x, bins=bins, density=True, histtype="step", color=color, lw=1.6, label=label)
    a2.axvline(0.6, color=INK3, lw=1, ls=(0, (3, 3)))
    a2.set_xlabel("face_recognition distance")
    a2.set_yticks([])
    a2.set_title("Distance to the enrolled photos")
    a2.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), fontsize=8.5, ncol=2)
    fig.text(0.012, -0.2, f"LFW: {m['enrolled_people']} people enrolled with {m['enrol_photos_per_person']} photos each, "
             f"{m['known_queries']} queries of enrolled people and {m['stranger_queries']} of strangers. "
             "Queries go through the app's 0.5 scale, HOG detector and matching rule.",
             color=INK3, fontsize=8.5, ha="left", va="top")
    fig.subplots_adjust(wspace=0.18)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "lfw_tolerance.png"), bbox_inches="tight", pad_inches=0.25)
    print("wrote docs/assets/lfw_tolerance.png")


if __name__ == "__main__":
    main()
