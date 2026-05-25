#!/usr/bin/env python3
"""Conformer relative energy correlation: Ref(eV) vs GFN2-xTB / g-xTB (Hartree)."""

import os, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

HARTREE_TO_KCAL = 627.5094740631
EV_TO_KCAL      = 23.060541945
BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)

MOLECULES = {
    "01_6_0109_n1": -1,
    "02_6_0901_0":   0,
    "03_6_0978_0":   0,
    "04_7_0231_n2": -2,
    "05_5_0513_0":   0,
    "06_7_0793_0":   0,
    "07_7_0083_0":   0,
    "08_7_0366_0":   0,
    "09_5_0584_0":   0,
    "10_6_0573_0":   0,
}

COLOR_GFN2 = "#2196F3"
COLOR_GXTB = "#E91E63"


# ── data loading ──────────────────────────────────────────────────────────────

def parse_ref_energy(xyz_path):
    with open(xyz_path) as f:
        f.readline()
        m = re.search(r"energy=([-\d.]+)", f.readline())
    return float(m.group(1)) if m else None

def read_energy(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        v = f.read().strip()
    return float(v) if v else None

def collect(mol_dir):
    xyz_dir = os.path.join(BASE, mol_dir)
    res_dir = os.path.join(xyz_dir, "xtb_results")
    ref, gfn2, gxtb = [], [], []
    for snap in sorted(f for f in os.listdir(xyz_dir) if f.startswith("snap_") and f.endswith(".xyz")):
        base = snap[:-4]
        er = parse_ref_energy(os.path.join(xyz_dir, snap))
        e2 = read_energy(os.path.join(res_dir, base, "gfn2.energy"))
        eg = read_energy(os.path.join(res_dir, base, "gxtb.energy"))
        if None not in (er, e2, eg):
            ref.append(er); gfn2.append(e2); gxtb.append(eg)
    ra, g2, gg = np.array(ref), np.array(gfn2), np.array(gxtb)
    return ((ra - ra.min()) * EV_TO_KCAL,
            (g2 - g2.min()) * HARTREE_TO_KCAL,
            (gg - gg.min()) * HARTREE_TO_KCAL,
            len(ref))


# ── per-subplot renderer ───────────────────────────────────────────────────────

def draw(ax, x, y, color, xlabel, ylabel, title):
    pr, _ = stats.pearsonr(x, y)
    sr, _ = stats.spearmanr(x, y)
    rmse  = np.sqrt(np.mean((x - y) ** 2))
    mae   = np.mean(np.abs(x - y))

    ax.scatter(x, y, s=12, alpha=0.45, color=color, edgecolors="none", zorder=3)
    lim = [min(x.min(), y.min()) - 0.3, max(x.max(), y.max()) + 0.3]
    ax.plot(lim, lim, "k--", lw=0.7, alpha=0.35, zorder=2)
    slope, intercept = stats.linregress(x, y)[:2]
    xf = np.linspace(lim[0], lim[1], 300)
    ax.plot(xf, slope * xf + intercept, color=color, lw=1.3, zorder=4)

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, pad=4)
    ax.tick_params(labelsize=7)

    txt = (f"r = {pr:.3f}   ρ = {sr:.3f}\n"
           f"RMSE = {rmse:.2f}  MAE = {mae:.2f} kcal/mol")
    ax.text(0.04, 0.97, txt, transform=ax.transAxes, fontsize=7,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))
    return pr, sr, rmse, mae


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    summary = []

    for mol_dir, charge in MOLECULES.items():
        ref_rel, gfn2_rel, gxtb_rel, n = collect(mol_dir)

        # One figure per molecule: 1 row × 2 square subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
        fig.suptitle(f"{mol_dir}   charge={charge:+d}   n={n}", fontsize=10)

        s2 = draw(ax1, ref_rel, gfn2_rel, COLOR_GFN2,
                  "Ref ΔE (kcal/mol)", "GFN2-xTB ΔE (kcal/mol)", "Ref vs GFN2-xTB")
        sg = draw(ax2, ref_rel, gxtb_rel, COLOR_GXTB,
                  "Ref ΔE (kcal/mol)", "g-xTB ΔE (kcal/mol)", "Ref vs g-xTB")

        fig.tight_layout()
        path = os.path.join(OUT, f"{mol_dir}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

        summary.append((mol_dir, charge, n, *s2, *sg))

    # ── summary figure: all molecules, colour-coded ──────────────────────────
    cmap = plt.colormaps["tab10"]
    mol_list = list(MOLECULES.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.suptitle("All molecules — conformer relative energies (kcal/mol)", fontsize=10)

    all_x, all_y2, all_yg = [], [], []

    for i, (mol_dir, charge) in enumerate(MOLECULES.items()):
        ref_rel, gfn2_rel, gxtb_rel, _ = collect(mol_dir)
        c = cmap(i / len(MOLECULES))
        ax1.scatter(ref_rel, gfn2_rel, s=10, alpha=0.4, color=c, edgecolors="none", zorder=3)
        ax2.scatter(ref_rel, gxtb_rel, s=10, alpha=0.4, color=c, edgecolors="none", zorder=3)
        all_x.append(ref_rel); all_y2.append(gfn2_rel); all_yg.append(gxtb_rel)

    all_x  = np.concatenate(all_x)
    all_y2 = np.concatenate(all_y2)
    all_yg = np.concatenate(all_yg)

    for ax, y, color, method in [
        (ax1, all_y2, COLOR_GFN2, "GFN2-xTB"),
        (ax2, all_yg, COLOR_GXTB, "g-xTB"),
    ]:
        lim = [min(all_x.min(), y.min()) - 0.3, max(all_x.max(), y.max()) + 0.3]
        ax.plot(lim, lim, "k--", lw=0.7, alpha=0.35, zorder=2)
        slope, intercept = stats.linregress(all_x, y)[:2]
        xf = np.linspace(lim[0], lim[1], 300)
        ax.plot(xf, slope * xf + intercept, color=color, lw=1.5, zorder=4)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Ref ΔE (kcal/mol)", fontsize=8)
        ax.set_ylabel(f"{method} ΔE (kcal/mol)", fontsize=8)
        ax.set_title(f"Ref vs {method} (all systems)", fontsize=9, pad=4)
        ax.tick_params(labelsize=7)
        pr, _ = stats.pearsonr(all_x, y)
        sr, _ = stats.spearmanr(all_x, y)
        rmse  = np.sqrt(np.mean((all_x - y) ** 2))
        mae   = np.mean(np.abs(all_x - y))
        txt = (f"r = {pr:.3f}   ρ = {sr:.3f}\n"
               f"RMSE = {rmse:.2f}  MAE = {mae:.2f} kcal/mol\n"
               f"N = {len(all_x)}")
        ax.text(0.04, 0.97, txt, transform=ax.transAxes, fontsize=7,
                va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))

    handles = [Line2D([0], [0], marker="o", color="none",
                      markerfacecolor=cmap(i / len(mol_list)), markersize=6,
                      label=mol_list[i])
               for i in range(len(mol_list))]
    fig.legend(handles=handles, fontsize=7, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    path = os.path.join(OUT, "summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # ── print table ──────────────────────────────────────────────────────────
    hdr = (f"{'Molecule':<18} {'q':>3} {'n':>4} │ "
           f"{'GFN2 r':>7} {'GFN2 ρ':>7} {'RMSE':>6} {'MAE':>6} │ "
           f"{'gxTB r':>7} {'gxTB ρ':>7} {'RMSE':>6} {'MAE':>6}")
    sep = "─" * len(hdr)
    print(f"\n{hdr}\n{sep}")
    for row in summary:
        mol, q, n, pr2, sr2, rm2, ma2, prg, srg, rmg, mag = row
        print(f"{mol:<18} {q:>+3} {n:>4} │ "
              f"{pr2:>7.4f} {sr2:>7.4f} {rm2:>6.2f} {ma2:>6.2f} │ "
              f"{prg:>7.4f} {srg:>7.4f} {rmg:>6.2f} {mag:>6.2f}")
    arr = np.array([[r[3],r[4],r[5],r[6],r[7],r[8],r[9],r[10]] for r in summary])
    m = arr.mean(0)
    print(sep)
    print(f"{'Mean':<18} {'':>3} {'':>4} │ "
          f"{m[0]:>7.4f} {m[1]:>7.4f} {m[2]:>6.2f} {m[3]:>6.2f} │ "
          f"{m[4]:>7.4f} {m[5]:>7.4f} {m[6]:>6.2f} {m[7]:>6.2f}")


if __name__ == "__main__":
    main()
