import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# --- Raw scaled similarities --------------------------------------------------
clean_similarities_noalign = [
    0.4338, 
    0.5514, 
    0.6112, 
    0.5938, 
    0.6224, 
    0.6675, 
    0.6533, 
    0.6617, 
    0.7046, 
    0.7126, 
    0.6390, 
    0.6923, 
    0.8141, 
    0.6883, 
    0.7138, 
    0.6491, 
    0.7413, 
    0.7385, 
    0.6353, 
    0.7847, 
    0.7512, 
    0.7422, 
    0.7509, 
    0.7516, 
    0.7339, 
    0.7696, 
    0.7581, 
    0.7313, 
    0.7431, 
    0.7597  
]
no_align_corrupt_sim = [
    0.3733,     
    0.4301,     
    0.4582, 
    0.4125,  
    0.4269,  
    0.4483,  
    0.4252, 
    0.4323, 
    0.4266,  
    0.4232,  
    0.3570,  
    0.3970,  
    0.5086, 
    0.3792, 
    0.3735, 
    0.3309, 
    0.3840, 
    0.4045, 
    0.3048, 
    0.4287, 
    0.3829, 
    0.3992, 
    0.3682, 
    0.3610, 
    0.3681, 
    0.3922, 
    0.3915,  
    0.3465,  
    0.3528,  
    0.3753   
]
with_align_clean_sim = [
    0.4324, 
    0.5789, 
    0.6428, 
    0.6012, 
    0.6801, 
    0.6862, 
    0.6816, 
    0.6924, 
    0.6940, 
    0.6218, 
    0.7388, 
    0.7108, 
    0.7465, 
    0.7379, 
    0.7915, 
    0.7641, 
    0.7292, 
    0.7238, 
    0.6730, 
    0.7354, 
    0.7427, 
    0.7340, 
    0.7395, 
    0.7402, 
    0.7664, 
    0.7106, 
    0.7394, 
    0.7722, 
    0.7754, 
    0.7799  
]
with_align_corrupt_sim = [
    0.3555, 
    0.4373, 
    0.4705, 
    0.4169, 
    0.4744, 
    0.4718, 
    0.4603, 
    0.4105, 
    0.4796, 
    0.3554, 
    0.4349, 
    0.4176, 
    0.4206, 
    0.4141, 
    0.4512, 
    0.3992, 
    0.3582, 
    0.3340, 
    0.3189, 
    0.3733, 
    0.3698, 
    0.3449, 
    0.3474, 
    0.3603, 
    0.3733, 
    0.3427, 
    0.3443, 
    0.3613, 
    0.3646, 
    0.3714  
]


print(f"No Align Clean: {np.array(clean_similarities_noalign)}")
print(f"No Align Corrupt: {np.array(no_align_corrupt_sim)}")
print(f"With Align Clean: {np.array(with_align_clean_sim)}")
print(f"With Align Corrupt: {np.array(with_align_corrupt_sim)}")
gap_base = np.array(clean_similarities_noalign) - np.array(no_align_corrupt_sim)
gap_align = np.array(with_align_clean_sim) - np.array(with_align_corrupt_sim)

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

plain_no_clean = np.array(clean_similarities_noalign)
plain_no_corrupt = np.array(no_align_corrupt_sim)
plain_wi_clean = np.array(with_align_clean_sim)
plain_wi_corrupt = np.array(with_align_corrupt_sim)


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

plt.legend(handles=legend_elements, loc='upper left', ncol=1, fontsize=13)

plt.tight_layout()

file_path_four = "DUTCH.png"
plt.savefig(file_path_four, dpi=600)

file_path_four
