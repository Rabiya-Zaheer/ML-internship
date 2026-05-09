# ============================================================
#  Task 1: Exploring and Visualizing the Iris Dataset
#  DevelopersHub Corp – ML Internship
# ============================================================

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import gaussian_kde
import warnings

warnings.filterwarnings("ignore")
matplotlib.use("Agg")          # change to "TkAgg" or remove if running interactively

# ── 1. LOAD DATASET ─────────────────────────────────────────────────────────
# Option A – load via seaborn (requires internet)
# df = sns.load_dataset("iris")

# Option B – load via sklearn (always available, no internet needed)
from sklearn.datasets import load_iris

iris_raw = load_iris()
df = pd.DataFrame(iris_raw.data, columns=["sepal_length", "sepal_width",
                                          "petal_length", "petal_width"])
df["species"] = pd.Categorical.from_codes(iris_raw.target, iris_raw.target_names)

# ── 2. BASIC INSPECTION ──────────────────────────────────────────────────────
print("=" * 55)
print("  IRIS DATASET – BASIC INSPECTION")
print("=" * 55)

print(f"\n📐 Shape : {df.shape}  ({df.shape[0]} rows × {df.shape[1]} columns)")

print("\n📋 Column names:")
print("  ", df.columns.tolist())

print("\n👀 First 5 rows (.head()):")
print(df.head().to_string(index=False))

print("\n🔍 Dataset info (.info()):")
df.info()

print("\n📊 Descriptive statistics (.describe()):")
print(df.describe().round(2))

print("\n🌸 Species value counts:")
print(df["species"].value_counts())

# ── 3. STYLE CONFIGURATION ───────────────────────────────────────────────────
PALETTE  = {"setosa": "#4ECDC4", "versicolor": "#FF6B6B", "virginica": "#45B7D1"}
BG       = "#0F1117"
CARD     = "#1A1D27"
TEXT     = "#E8E8F0"
MUTED    = "#6B7280"
ACCENT   = "#F5D547"

features    = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
feat_labels = ["Sepal Length",  "Sepal Width",  "Petal Length",  "Petal Width"]
species_list = list(PALETTE.keys())


def style_ax(ax):
    """Apply dark-theme styling to an axis."""
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_color("#2A2D3A")
    ax.tick_params(colors=MUTED)


# ── 4. SCATTER PLOT MATRIX ───────────────────────────────────────────────────
print("\n⏳ Generating scatter plot matrix …")

fig1, axes = plt.subplots(4, 4, figsize=(16, 14))
fig1.patch.set_facecolor(BG)
fig1.suptitle("Iris Dataset — Scatter Plot Matrix",
              color=TEXT, fontsize=20, fontweight="bold", y=1.01)

for i, (fy, ly) in enumerate(zip(features, feat_labels)):
    for j, (fx, lx) in enumerate(zip(features, feat_labels)):
        ax = axes[i][j]
        style_ax(ax)
        ax.tick_params(labelsize=7)

        if i == j:                          # diagonal → histogram per species
            for sp, col in PALETTE.items():
                ax.hist(df[df["species"] == sp][fx], bins=15,
                        color=col, alpha=0.7, edgecolor="none")
        else:                               # off-diagonal → scatter
            for sp, col in PALETTE.items():
                sub = df[df["species"] == sp]
                ax.scatter(sub[fx], sub[fy], color=col,
                           alpha=0.7, s=18, edgecolors="none")

        if i == 3:
            ax.set_xlabel(lx, color=TEXT, fontsize=8)
        if j == 0:
            ax.set_ylabel(ly, color=TEXT, fontsize=8)

legend_patches = [mpatches.Patch(color=c, label=s.capitalize())
                  for s, c in PALETTE.items()]
fig1.legend(handles=legend_patches, loc="lower center", ncol=3,
            frameon=False, fontsize=11, labelcolor=TEXT,
            bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.03, 1, 1])
fig1.savefig("fig1_scatter_matrix.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✅  Saved → fig1_scatter_matrix.png")


# ── 5. HISTOGRAMS ────────────────────────────────────────────────────────────
print("⏳ Generating histograms …")

fig2, axes = plt.subplots(2, 2, figsize=(14, 9))
fig2.patch.set_facecolor(BG)
fig2.suptitle("Iris Dataset — Feature Distributions (Histograms)",
              color=TEXT, fontsize=18, fontweight="bold", y=1.02)

for ax, feat, label in zip(axes.flat, features, feat_labels):
    style_ax(ax)
    for sp, col in PALETTE.items():
        subset = df[df["species"] == sp][feat]
        ax.hist(subset, bins=18, color=col, alpha=0.75,
                edgecolor="none", label=sp.capitalize())
        # Overlay smooth KDE curve
        xs  = np.linspace(subset.min() - 0.5, subset.max() + 0.5, 200)
        kde = gaussian_kde(subset)
        ax.plot(xs,
                kde(xs) * len(subset) * (subset.max() - subset.min()) / 18,
                color=col, linewidth=2)

    ax.set_title(label, color=TEXT, fontsize=12, fontweight="bold")
    ax.set_xlabel("Value (cm)", color=MUTED, fontsize=9)
    ax.set_ylabel("Count",      color=MUTED, fontsize=9)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9)
    ax.grid(axis="y", color="#2A2D3A", linewidth=0.5)

plt.tight_layout()
fig2.savefig("fig2_histograms.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✅  Saved → fig2_histograms.png")


# ── 6. BOX PLOTS ─────────────────────────────────────────────────────────────
print("⏳ Generating box plots …")

fig3, axes = plt.subplots(1, 4, figsize=(16, 7))
fig3.patch.set_facecolor(BG)
fig3.suptitle("Iris Dataset — Box Plots (Outlier Detection)",
              color=TEXT, fontsize=18, fontweight="bold", y=1.02)

for ax, feat, label in zip(axes, features, feat_labels):
    style_ax(ax)
    data_groups = [df[df["species"] == sp][feat].values for sp in species_list]

    bp = ax.boxplot(
        data_groups,
        patch_artist=True,
        medianprops  =dict(color=ACCENT, linewidth=2.5),
        whiskerprops =dict(color=MUTED,  linewidth=1.2),
        capprops     =dict(color=MUTED,  linewidth=1.5),
        flierprops   =dict(marker="o", markerfacecolor=ACCENT,
                           markeredgecolor="none", markersize=5, alpha=0.8),
    )
    for patch, col in zip(bp["boxes"], PALETTE.values()):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([s.capitalize() for s in species_list],
                       color=TEXT, fontsize=9, rotation=12)
    ax.set_title(label,      color=TEXT,  fontsize=12, fontweight="bold")
    ax.set_ylabel("Value (cm)", color=MUTED, fontsize=9)
    ax.grid(axis="y", color="#2A2D3A", linewidth=0.5)

plt.tight_layout()
fig3.savefig("fig3_boxplots.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("   ✅  Saved → fig3_boxplots.png")


# ── 7. DONE ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ALL DONE!  Three figures saved successfully.")
print("=" * 55)
print("""
  fig1_scatter_matrix.png  →  scatter plots across all feature pairs
  fig2_histograms.png      →  distribution of each feature per species
  fig3_boxplots.png        →  box plots showing spread & outliers
""")
