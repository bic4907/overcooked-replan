"""Figure 2: the advance alert on the unstructured (V1, V2) and structured
(OvercookedV3) transitions, alert OFF beside alert ON.

Each cell is one kitchen. Its cross-play return under an arm is normalized by
the best score seen in that cell under either arm, self-play included in the
reference only, so a bar reads as "share of the cell's best". Bars are the
mean over a group's cells and error bars the 95% interval of the paired
OFF-ON comparison, taken within cells so that overlap reads as the paired
test does.
The axis starts at 40% rather than zero, so bar heights are not proportional
to the values; the paired one-sided t-test (ON > OFF) is printed per group
rather than drawn.

The numbers come from the six-seed evaluation projects and are cached in
``outputs/figures/alert_groups_cells.csv``; ``--refresh`` pulls them again.
Two versions are written into ``results/paper``: the bars alone
(``alert_groups``) and the bars with one dot per cell and the OFF-ON pairs
joined (``alert_groups_dots``).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "paper"
SOURCE_CSV = ROOT / "outputs/figures/alert_groups_cells.csv"
ENTITY = "cilab-overcooked"

#: group -> arm -> evaluation project. The structured group's two arms are the
#: 0921 six-kitchen campaign, alert on and alert off; distance_0 was retrained
#: and re-scored in both arms on 09-26 after that kitchen was redrawn, and only
#: the newest run per kitchen counts.
PROJECTS = {
    "V1": {"off": "overcooked-v1map-ippo-cdoff-rnn-0916_eval", "on": "overcooked-v1map-ippo-cdon-rnn-0916_eval"},
    "V2": {"off": "overcooked-v2map-ippo-cdoff-rnn-0918_eval", "on": "overcooked-v2map-ippo-cdon-rnn-0918_eval"},
    "Structured": {"off": "overcooked-v3-ippo-rnn-observer-none-0921_test", "on": "overcooked-v3-ippo-rnn-0921_eval"},
}
#: No kitchen needs a project of its own any more.
OVERRIDES = {}
#: Where a ten-seed evaluation of the same arm exists, it is used instead: the
#: cross-play mean of a cell moves by up to a hundred points between six and
#: ten seeds even though its self-play barely moves, because cross-play
#: averages ordered pairs and four more seeds bring sixty more of them.
TEN_SEED = {
    "overcooked-v1map-ippo-cdoff-rnn-0916_eval": "overcooked-v1map-ippo-cdoff-rnn-0916_eval10",
    "overcooked-v1map-ippo-cdon-rnn-0916_eval": "overcooked-v1map-ippo-cdon-rnn-0916_eval10",
    "overcooked-v2map-ippo-cdoff-rnn-0918_eval": "overcooked-v2map-ippo-cdoff-rnn-0918_eval10",
    "overcooked-v2map-ippo-cdon-rnn-0918_eval": "overcooked-v2map-ippo-cdon-rnn-0918_eval10",
    "overcooked-v3-ippo-rnn-observer-none-0921_test": "overcooked-v3-ippo-rnn-observer-none-0921_test10",
}
STRUCTURED = ("split_0", "split_1", "outage_0", "outage_1", "distance_0", "distance_1")
SEEDS = 6


def fetch_cells():
    import wandb

    api = wandb.Api()
    cells = []
    for group, arms in PROJECTS.items():
        for arm, project in arms.items():
            ten = TEN_SEED.get(project)
            if ten is not None:
                try:
                    if any(run.state == "finished" for run in api.runs(f"{ENTITY}/{ten}", per_page=5)):
                        print(f"  {group}/{arm}: using the ten-seed evaluation {ten}")
                        project = ten
                except Exception:
                    pass
            newest = {}
            for run in api.runs(f"{ENTITY}/{project}", per_page=100):
                if run.state != "finished" or "_old" in run.name:
                    continue
                layout = str(run.config.get("layout"))
                if group == "Structured" and layout not in STRUCTURED:
                    continue
                if layout not in newest or run.created_at > newest[layout].created_at:
                    newest[layout] = run
            for (g, a, layout), project_ in OVERRIDES.items():
                if (g, a) != (group, arm):
                    continue
                for run in api.runs(f"{ENTITY}/{project_}", per_page=100):
                    if run.state == "finished" and str(run.config.get("layout")) == layout and not run.config.get("stochastic"):
                        if layout not in newest or "0920" not in newest[layout].project or run.created_at > newest[layout].created_at:
                            newest[layout] = run
            for layout, run in newest.items():
                s = run.summary
                cells.append(dict(
                    group=group, arm=arm, layout=layout, sp=float(s["SP"]), xp=float(s["XP"]),
                    sp_pairs=int(s.get("counts/SP_pairs") or 0), xp_drop=float(s.get("XP/adaptation/drop", "nan")),
                    run=f"{ENTITY}/{project}/{run.id}",
                ))
    return cells


def write_cells(cells, path=SOURCE_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0].keys()))
        writer.writeheader()
        writer.writerows(cells)


def load_cells(path=SOURCE_CSV):
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row, sp=float(row["sp"]), xp=float(row["xp"]), sp_pairs=int(row["sp_pairs"]), xp_drop=float(row["xp_drop"])) for row in csv.DictReader(stream)]


def normalized(cells):
    """group -> (layouts, off, on, off_sp, on_sp), normalized, cells with both arms.

    Self-play is carried alongside cross-play, divided by the same per-cell
    reference, for the companion self-play figure.
    """
    out = {}
    for group in PROJECTS:
        mine = {(c["layout"], c["arm"]): c for c in cells if c["group"] == group}
        layouts = sorted({l for l, _ in mine} & {l for l, a in mine if a == "on"} & {l for l, a in mine if a == "off"})
        off, on, off_sp, on_sp = [], [], [], []
        for l in layouts:
            a, b = mine[(l, "off")], mine[(l, "on")]
            best = max(a["sp"], a["xp"], b["sp"], b["xp"])
            off.append(a["xp"] / best if best > 0 else np.nan)
            on.append(b["xp"] / best if best > 0 else np.nan)
            off_sp.append(a["sp"] / best if best > 0 else np.nan)
            on_sp.append(b["sp"] / best if best > 0 else np.nan)
            if a["sp_pairs"] < SEEDS or b["sp_pairs"] < SEEDS:
                print(f"  warning: {group}/{l} evaluated on {a['sp_pairs']}/{b['sp_pairs']} seeds")
        out[group] = (layouts, np.array(off), np.array(on), np.array(off_sp), np.array(on_sp))
    return out


def ci95_within(off, on):
    """Per-arm 95% interval for the paired OFF-ON comparison, one per arm.

    The cells differ enormously among themselves - a kitchen scores anywhere
    from a quarter to all of its best - so an interval taken across cells is
    dominated by that spread rather than by the contrast the figure is about.
    Removing each cell's own level first (Cousineau, with Morey's sqrt(J/(J-1))
    correction for J=2 arms) leaves intervals whose overlap reads as the paired
    test does, matching the paired intervals used elsewhere in the paper.
    """
    off, on = np.asarray(off, float), np.asarray(on, float)
    n = int(np.sum(~np.isnan(off) & ~np.isnan(on)))
    if n < 2:
        return 0.0, 0.0
    centred = np.nanmean(np.vstack([off, on]), axis=0)
    grand = np.nanmean(np.hstack([off, on]))
    factor = np.sqrt(2.0) * stats.t.ppf(0.975, n - 1)
    return tuple(float(factor * stats.sem(arm - centred + grand, nan_policy="omit"))
                 for arm in (off, on))


#: Point sizes shared with the companion figure alert_preparation_barplot.
SIZES = {"tick": 8.0, "value": 7.2, "axis": 8.5, "legend": 8.5}
#: What each group's bars are called on the axis.
#: Groups are named by the work they come from rather than by version numbers
#: that were never theirs, and each by how its edits are placed: at random, or
#: aimed at the structure the division of work rests on.
LABELS = {"V1": "Carroll et al.,\nrandom edit", "V2": "Gessler et al.,\nrandom edit",
          "Structured": "Allocation-aware\nedit"}


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


#: The two arms' colours in the paper figure until now; ``draw`` takes others.
ARM_COLORS = {"off": "#b9c0c7", "on": "#1f7a8c"}


def draw(groups, dots, stem=None, ylim=(0.4, 1.0), width_in=3.55, marks=False,
         colors=None, out_dir=None, xlim=None, right_in=0.053, legend_below=False,
         arm_labels=("Alert OFF", "Alert ON"), tick_labels=None, extra_bottom=0.0,
         hatch_off=None, bar_width=0.34, show_values=True, legend_kw=None, tick_size=None, ytick_size=None, height_in=None):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                         "font.size": 8.0, "pdf.fonttype": 42, "ps.fonttype": 42})
    # Significance brackets stand above the value labels, so a marked figure is
    # a little taller and lifts its legend clear of them.
    height_in = height_in or (2.15 if marks else 1.95)
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    # A key under the axis labels instead of over the plot: the plot keeps the
    # height the key took on top, and the foot grows by the key's row.
    key_h = 0.22 if legend_below else 0.0
    colors = colors or ARM_COLORS
    width = bar_width
    print(f"{'group':11s} {'n':>3s} {'OFF':>6s} {'±95%':>6s} {'ON':>6s} {'±95%':>6s} {'diff':>7s} {'95% CI of diff':>17s} {'t':>6s} {'p':>8s}")
    for i, (group, (layouts, off, on, _, _)) in enumerate(groups.items()):
        halves = ci95_within(off, on)
        for k, (arm, values) in enumerate((("off", off), ("on", on))):
            x = i + (k - 0.5) * width
            mean, half = np.nanmean(values), halves[k]
            if hatch_off and arm == "off":
                # The arm without the alert: the same colour, hatched.
                ax.bar(x, mean, width=width * 0.92, facecolor=colors["on"], edgecolor="white",
                       hatch=hatch_off, linewidth=0, zorder=2)
            else:
                ax.bar(x, mean, width=width * 0.92, color=colors[arm], zorder=2)
            ax.errorbar(x, mean, yerr=half, fmt="none", ecolor="#333", lw=0.8, capsize=2, zorder=3)
            if show_values:
                ax.text(x, mean + half + 0.012, f"{mean * 100:.1f}%", ha="center", va="bottom", fontsize=SIZES["value"], fontweight="bold", color="#1a1a1a", zorder=6)
        if dots:
            # One dot per cell on each bar, and a hairline joining a cell's
            # two arms, so the paired differences the test is run on show.
            jitter = (np.arange(len(layouts)) - (len(layouts) - 1) / 2) * (width * 0.7 / max(len(layouts), 1))
            x_off, x_on = i - width / 2 + jitter, i + width / 2 + jitter
            for xo, xn, a, b in zip(x_off, x_on, off, on):
                ax.plot([xo, xn], [a, b], color="#4a525b", lw=0.4, alpha=0.6, zorder=4)
            ax.scatter(x_off, off, s=7, facecolor="white", edgecolor="#4a525b", lw=0.5, zorder=5)
            ax.scatter(x_on, on, s=7, facecolor="white", edgecolor="#4a525b", lw=0.5, zorder=5)
        t, p = stats.ttest_rel(on, off, nan_policy="omit", alternative="greater")
        if marks:
            # Two-sided, so a mark agrees with whether the 95% interval of the
            # paired difference drawn on the bars excludes zero.
            p2 = stats.ttest_rel(on, off, nan_policy="omit").pvalue
            top = max(np.nanmean(off) + halves[0], np.nanmean(on) + halves[1]) + 0.062
            ax.plot([i - width / 2, i - width / 2, i + width / 2, i + width / 2],
                    [top, top + 0.012, top + 0.012, top], color="#333", lw=0.7, clip_on=False)
            ax.text(i, top + 0.018, stars(p2), ha="center", va="bottom",
                    fontsize=SIZES["value"], color="#1a1a1a", clip_on=False)
        if not dots:
            d = on - off
            m = stats.t.ppf(0.975, len(layouts) - 1) * stats.sem(d, nan_policy="omit")
            print(f"{group:11s} {len(layouts):3d} {np.nanmean(off):6.3f} {halves[0]:6.3f} {np.nanmean(on):6.3f} {halves[1]:6.3f} {np.nanmean(d):+7.3f} [{np.nanmean(d)-m:+.3f},{np.nanmean(d)+m:+.3f}] {t:6.2f} {p:8.4f}")
            for l, a, b in zip(layouts, off, on):
                print(f"    {l:32s} off {a:.3f}  on {b:.3f}  Δ {b - a:+.3f}")
    ax.set_xticks(range(len(groups)), [(tick_labels or LABELS)[g] for g in groups],
                  fontsize=tick_size or SIZES["tick"])
    # The bars are read at 40-100%; the dot version keeps the full range so
    # that no cell falls off the axis.
    ax.set_ylim((0.0, 1.05) if dots else ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Normalized return (%)", fontsize=SIZES["axis"])
    ax.tick_params(labelsize=SIZES["tick"])
    if ytick_size:
        ax.tick_params(axis="y", labelsize=ytick_size)
    ax.grid(axis="y", color="#e6e9ec", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    key = [plt.Rectangle((0, 0), 1, 1, facecolor=colors["on"], edgecolor="white", hatch="///",
                         linewidth=0, label=arm_labels[0]) if hatch_off else
           plt.Rectangle((0, 0), 1, 1, color=colors["off"], label=arm_labels[0]),
           plt.Rectangle((0, 0), 1, 1, color=colors["on"], label=arm_labels[1])]
    if legend_below:
        # Long swatches, so the hatching shows as it does on the bars.
        fig.legend(handles=key, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
                   **{**dict(fontsize=SIZES["legend"], columnspacing=0.9, handletextpad=0.4,
                             handlelength=2.2), **(legend_kw or {})})
    else:
        ax.legend(handles=key, loc="lower center", bbox_to_anchor=(0.5, 1.09 if marks else 1.0), ncol=2,
                  frameon=False, fontsize=SIZES["legend"], columnspacing=3.2, handletextpad=0.7,
                  handlelength=1.5)
    # Margins in inches, so a narrower figure keeps room for the axis label.
    top_in = (0.06 if legend_below else 0.234) + (0.16 if marks else 0)
    fig.subplots_adjust(left=0.586 / width_in, right=1 - right_in / width_in,
                        top=1 - top_in / height_in,
                        bottom=(0.458 + key_h + extra_bottom) / height_in)
    stem = stem or ("alert_groups_dots" if dots else "alert_groups")
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=240)
    plt.close(fig)
    print("figure ->", out_dir / f"{stem}.png")


def draw_combined(groups):
    """Cross-play in front of self-play, one pair of bars per arm.

    Same reading as the benchmark's other performance figures: the wide hatched
    bar is the arm's self-play, the solid bar in front is its cross-play, and
    the space between them is what is lost to an unfamiliar partner.
    """
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                         "font.size": 8.0, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.55, 1.95))
    colors = {"off": "#b9c0c7", "on": "#1f7a8c"}
    label_colors = {"off": "#1a1a1a", "on": "white"}
    width = 0.34
    for i, (group, (layouts, off, on, off_sp, on_sp)) in enumerate(groups.items()):
        halves = ci95_within(off, on)
        for k, (arm, xp, sp) in enumerate((("off", off, off_sp), ("on", on, on_sp))):
            x = i + (k - 0.5) * width
            mean, half = np.nanmean(xp), halves[k]
            ax.bar(x, np.nanmean(sp), width=width * 0.92, color="white",
                   edgecolor=colors[arm], linewidth=0.9, hatch="////", zorder=1)
            ax.bar(x, mean, width=width * 0.60, color=colors[arm],
                   edgecolor="white", linewidth=0.35, zorder=3)
            ax.errorbar(x, mean, yerr=half, fmt="none", ecolor="#333", lw=0.8, capsize=2, zorder=4)
            ax.text(x, mean - 0.035, f"{mean * 100:.0f}", ha="center", va="top",
                    fontsize=SIZES["value"], fontweight="bold", color=label_colors[arm], zorder=6)
    ax.set_xticks(range(len(groups)), [LABELS[g] for g in groups], fontsize=SIZES["tick"])
    ax.set_ylim(0.4, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Normalized return (%)", fontsize=SIZES["axis"])
    ax.tick_params(labelsize=SIZES["tick"])
    ax.grid(axis="y", color="#e6e9ec", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors["off"], label="Alert OFF"),
                       plt.Rectangle((0, 0), 1, 1, color=colors["on"], label="Alert ON"),
                       plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#4a525b", hatch="////", label="Self-play")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              fontsize=SIZES["legend"], columnspacing=1.5, handletextpad=0.5, handlelength=1.2)
    fig.subplots_adjust(left=0.165, right=0.985, top=0.88, bottom=0.235)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "alert_groups_spxp.pdf")
    fig.savefig(OUT_DIR / "alert_groups_spxp.png", dpi=240)
    plt.close(fig)
    print("figure ->", OUT_DIR / "alert_groups_spxp.png")


def draw_selfplay(groups):
    """The companion figure: self-play on the same normalized scale.

    Cross-play is what the alert is judged on, but a reader wants to know
    whether an arm simply learned a better policy. Self-play answers that, and
    it sits in its own figure so the main one keeps two bars per group.
    """
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                         "font.size": 8.0, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.55, 1.95))
    colors = {"off": "#b9c0c7", "on": "#1f7a8c"}
    width = 0.34
    print(f"{'group':11s} {'n':>3s} {'SP OFF':>7s} {'SP ON':>7s} {'diff':>7s} {'95% CI of diff':>17s}")
    for i, (group, (layouts, _, _, off_sp, on_sp)) in enumerate(groups.items()):
        halves = ci95_within(off_sp, on_sp)
        for k, (arm, values) in enumerate((("off", off_sp), ("on", on_sp))):
            x = i + (k - 0.5) * width
            mean, half = np.nanmean(values), halves[k]
            ax.bar(x, mean, width=width * 0.92, color=colors[arm], zorder=2)
            ax.errorbar(x, mean, yerr=half, fmt="none", ecolor="#333", lw=0.8, capsize=2, zorder=3)
            ax.text(x, mean + half + 0.012, f"{mean * 100:.1f}%", ha="center", va="bottom",
                    fontsize=SIZES["value"], fontweight="bold", color="#1a1a1a", zorder=6)
        d = on_sp - off_sp
        m = stats.t.ppf(0.975, len(layouts) - 1) * stats.sem(d, nan_policy="omit")
        print(f"{group:11s} {len(layouts):3d} {np.nanmean(off_sp):7.3f} {np.nanmean(on_sp):7.3f} "
              f"{np.nanmean(d):+7.3f} [{np.nanmean(d) - m:+.3f},{np.nanmean(d) + m:+.3f}]")
    ax.set_xticks(range(len(groups)), [LABELS[g] for g in groups], fontsize=SIZES["tick"])
    ax.set_ylim(0.4, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Normalized self-play (%)", fontsize=SIZES["axis"])
    ax.tick_params(labelsize=SIZES["tick"])
    ax.grid(axis="y", color="#e6e9ec", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors["off"], label="Alert OFF"),
                       plt.Rectangle((0, 0), 1, 1, color=colors["on"], label="Alert ON")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              fontsize=SIZES["legend"], columnspacing=3.2, handletextpad=0.7, handlelength=1.5)
    fig.subplots_adjust(left=0.165, right=0.985, top=0.88, bottom=0.235)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "alert_groups_selfplay.pdf")
    fig.savefig(OUT_DIR / "alert_groups_selfplay.png", dpi=240)
    plt.close(fig)
    print("figure ->", OUT_DIR / "alert_groups_selfplay.png")


def main():
    if "--refresh" in sys.argv or not SOURCE_CSV.exists():
        write_cells(fetch_cells())
    groups = normalized(load_cells())
    draw(groups, dots=False)
    draw(groups, dots=True)
    # Without the OvercookedV2 kitchens, whose cross-play sits at the ceiling,
    # the two remaining groups fit a 40-80% axis.
    draw({k: v for k, v in groups.items() if k != "V2"}, dots=False,
         stem="alert_groups_noV2", ylim=(0.4, 0.8), width_in=2.6, marks=True)
    draw_selfplay(groups)
    draw_combined(groups)


if __name__ == "__main__":
    main()
