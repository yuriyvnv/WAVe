import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# --- Raw scaled similarities --------------------------------------------------
no_align_clean_sim = [
    0.0474, 0.3295, 0.4569, 0.4935, 0.5942, 0.5773, 0.6094, 0.6054, 0.6282, 0.6409,
    0.6329, 0.6732, 0.6180, 0.6810, 0.6446, 0.7201, 0.6943, 0.7162, 0.7511, 0.6895,
    0.6436, 0.7292, 0.6585, 0.6216, 0.6491, 0.7093, 0.7226, 0.6821, 0.6681, 0.7543
]
no_align_corrupt_sim = [
    0.0339, 0.1992, 0.2749, 0.2809, 0.3597, 0.3289, 0.3416, 0.3324, 0.3381, 0.3338,
    0.3190, 0.3435, 0.2805, 0.3251, 0.2914, 0.3419, 0.3080, 0.3205, 0.3606, 0.2810,
    0.2365, 0.3020, 0.2528, 0.2012, 0.2283, 0.2811, 0.2710, 0.2303, 0.2448, 0.3034
]
with_align_clean_sim = [
    0.1485, 0.5403, 0.6453, 0.6493, 0.8196, 0.6200, 0.9188, 0.6682, 0.7884, 0.8915,
    0.8448, 0.8511, 0.8385, 0.8296, 0.7917, 0.9004, 0.7811, 0.9612, 0.8652, 0.8910,
    0.8433, 0.9276, 0.9523, 0.9326, 0.8787, 0.8957, 0.8074, 0.7844, 0.8746, 0.7371
]
with_align_corrupt_sim = [
    0.0914, 0.3360, 0.4012, 0.3713, 0.5286, 0.3285, 0.6100, 0.3007, 0.3944, 0.5113,
    0.4299, 0.4250, 0.3917, 0.4273, 0.3418, 0.4322, 0.2877, 0.5316, 0.3400, 0.3512,
    0.3093, 0.3863, 0.4336, 0.4124, 0.3013, 0.3398, 0.2591, 0.2361, 0.3010, 0.2115
]

# --- Inverse sigmoid to plain cosine 0–1 -------------------------------------
T = 0.1
def to_plain(arr):
    arr = np.asarray(arr)
    cos = T * np.log(arr / (1 - arr))
    return np.clip((cos + 1) / 2, 0, 1)
print(f"No Align Clean: {to_plain(no_align_clean_sim)}")
print(f"No Align Corrupt: {to_plain(no_align_corrupt_sim)}")
print(f"With Align Clean: {to_plain(with_align_clean_sim)}")
print(f"With Align Corrupt: {to_plain(with_align_corrupt_sim)}")
gap_base = to_plain(no_align_clean_sim) - to_plain(no_align_corrupt_sim)
gap_align = to_plain(with_align_clean_sim) - to_plain(with_align_corrupt_sim)

# --- Spline interpolation for smooth curve -----------------------------------
epochs = np.arange(1, 31)
epochs_smooth = np.linspace(1, 30, 300)

spline_base = make_interp_spline(epochs, gap_base, k=3)(epochs_smooth)
spline_align = make_interp_spline(epochs, gap_align, k=3)(epochs_smooth)

# --- Plot ---------------------------------------------------------------------
palette = {"baseline": "#6F826A", "aligned": "#BF9264", "fill": "#BBD8A3"}

plt.figure(figsize=(10, 6), dpi=120)

plt.plot(epochs_smooth, spline_base, color=palette["baseline"], linewidth=2.5,
         label="Baseline Gap")
plt.plot(epochs_smooth, spline_align, color=palette["aligned"], linewidth=3,
         label="Aligned Gap")

# X markers at raw points
plt.scatter(epochs, gap_base, marker='x', s=70, color=palette["baseline"])
plt.scatter(epochs, gap_align, marker='x', s=70, color=palette["aligned"])

# shaded region
plt.fill_between(epochs_smooth, spline_base, spline_align,
                 where=spline_align >= spline_base,
                 color=palette["fill"], alpha=0.25)

plt.title("Evolution Difference of the Similarity Gap", fontsize=18)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Similarity Gap (cosine 0–1)", fontsize=14)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

# Re‑plot second chart with new markers (distinct + small) and colors

epochs = np.arange(1, 31)
def spline_xy(x, y, pts=300):
    xs = np.linspace(x.min(), x.max(), pts)
    return xs, make_interp_spline(x, y, k=3)(xs)

plain_no_clean = to_plain(no_align_clean_sim)
plain_no_corrupt = to_plain(no_align_corrupt_sim)
plain_wi_clean = to_plain(with_align_clean_sim)
plain_wi_corrupt = to_plain(with_align_corrupt_sim)

x_s, y_nc = spline_xy(epochs, plain_no_clean)
_,   y_nk = spline_xy(epochs, plain_no_corrupt)
_,   y_wc = spline_xy(epochs, plain_wi_clean)
_,   y_wk = spline_xy(epochs, plain_wi_corrupt)

colors = {
    "no_clean": "#6F826A",
    "no_corrupt": "#E07A5F",
    "wi_clean": "#BF9264",
    "wi_corrupt": "#447D9B"
}
markers = {
    "no_clean": 'o',
    "no_corrupt": 's',
    "wi_clean": '^',
    "wi_corrupt": 'D'
}

plt.figure(figsize=(10,6), dpi=120)
plt.plot(x_s, y_nc, color=colors["no_clean"], linewidth=2, label="No‑Align Clean")
plt.plot(x_s, y_nk, color=colors["no_corrupt"], linewidth=2, label="No‑Align Corrupt")
plt.plot(x_s, y_wc, color=colors["wi_clean"], linewidth=2, label="With‑Align Clean")
plt.plot(x_s, y_wk, color=colors["wi_corrupt"], linewidth=2, label="With‑Align Corrupt")

plt.scatter(epochs, plain_no_clean, marker=markers["no_clean"], s=40, color=colors["no_clean"])
plt.scatter(epochs, plain_no_corrupt, marker=markers["no_corrupt"], s=40, color=colors["no_corrupt"])
plt.scatter(epochs, plain_wi_clean, marker=markers["wi_clean"], s=40, color=colors["wi_clean"])
plt.scatter(epochs, plain_wi_corrupt, marker=markers["wi_corrupt"], s=40, color=colors["wi_corrupt"])
# Step 1: Calculate max gap indices and values
gap_no_align = np.abs(plain_no_clean - plain_no_corrupt)
gap_with_align = np.abs(plain_wi_clean - plain_wi_corrupt)

idx_no_align = np.argmax(gap_no_align)
idx_with_align = np.argmax(gap_with_align)

epoch_no_align = epochs[idx_no_align]
epoch_with_align = epochs[idx_with_align]

# Draw vertical lines for both max gaps (within curve bounds)
plt.vlines(epoch_with_align,
           ymin=min(plain_wi_clean[idx_with_align], plain_wi_corrupt[idx_with_align]),
           ymax=max(plain_wi_clean[idx_with_align], plain_wi_corrupt[idx_with_align]),
           color='black', linestyle='--', linewidth=2)

plt.vlines(epoch_no_align,
           ymin=min(plain_no_clean[idx_no_align], plain_no_corrupt[idx_no_align]),
           ymax=max(plain_no_clean[idx_no_align], plain_no_corrupt[idx_no_align]),
           color='black', linestyle='-.', linewidth=2)

# Label on the red dashed line (with alignment)
y_middle_with = (plain_wi_clean[idx_with_align] + plain_wi_corrupt[idx_with_align]) / 2
plt.text(epoch_with_align + 0.4, y_middle_with,
         f'Max Gap\n({gap_with_align[idx_with_align]:.4f})\nW/ Align',
         color='black', fontsize=12, va='center', ha='left')

# Label on the purple dashed line (no alignment)
y_middle_no = (plain_no_clean[idx_no_align] + plain_no_corrupt[idx_no_align]) / 2
plt.text(epoch_no_align + 0.4, y_middle_no,
         f'Max Gap\n({gap_no_align[idx_no_align]:.4f})\nW/o Align',
         color='black', fontsize=12, va='center', ha='left')



plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Similarity (0–1)", fontsize=18)
plt.grid(alpha=0.25)
from matplotlib.lines import Line2D

# Custom legend handles with color + marker
legend_elements = [
    Line2D([0], [0], color=colors["no_clean"], marker='o', label='No‑Align Clean', markersize=7, linestyle='-'),
    Line2D([0], [0], color=colors["no_corrupt"], marker='s', label='No‑Align Corrupt', markersize=7, linestyle='-'),
    Line2D([0], [0], color=colors["wi_clean"], marker='^', label='With‑Align Clean', markersize=7, linestyle='-'),
    Line2D([0], [0], color=colors["wi_corrupt"], marker='D', label='With‑Align Corrupt', markersize=7, linestyle='-')
]

plt.legend(handles=legend_elements, loc='upper left', ncol=2, fontsize=13)

plt.tight_layout()

file_path_four = "PORTUGUESE.png"
plt.savefig(file_path_four, dpi=600)

file_path_four
