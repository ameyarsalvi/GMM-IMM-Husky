import json
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path

# --- Load global model parameters ---
with open("model_params.json", "r") as f:
    global_model = json.load(f)

A_global = global_model["A"]
B1_global = global_model["B1"]
B2_global = global_model["B2"]

# --- Get all dense segment files ---
segment_dir = Path("local_model_segments_dense")
segment_files = list(segment_dir.glob("segment_*.json"))

# --- Choose 6 random segments ---
random_segments = random.sample(segment_files, 6)

# --- Setup 3x2 subplot grid ---
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

# --- Loop through selected segments ---
for idx, seg_path in enumerate(random_segments):
    with open(seg_path, "r") as f:
        segment = json.load(f)

    # Extract data
    omega_k = np.array(segment["omega"])
    omega_k1 = np.array(segment["omega_next"])
    phiDotL = np.array(segment["phiDotL"])
    phiDotR = np.array(segment["phiDotR"])

    # Local model prediction
    A_local = segment["A"]
    B1_local = segment["B1"]
    B2_local = segment["B2"]

    omega_pred_local = A_local * omega_k + B1_local * phiDotL + B2_local * phiDotR
    omega_pred_global = A_global * omega_k + B1_global * phiDotL + B2_global * phiDotR

    # RMSE
    rmse_local = np.sqrt(np.mean((omega_k1 - omega_pred_local)**2))
    rmse_global = np.sqrt(np.mean((omega_k1 - omega_pred_global)**2))

    # Plot
    ax = axes[idx]
    ax.plot(omega_k1, 'k', linewidth=2, label="True $\\omega_{k+1}$")
    ax.plot(omega_pred_local, 'g--', linewidth=2, label="Local")
    ax.plot(omega_pred_global, 'r--', linewidth=2, label="Global")
    ax.set_title(f"Segment {segment['segment_index']}\nRMSE Local: {rmse_local:.4f}, Global: {rmse_global:.4f}")
    ax.grid(True)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("$\\omega_{k+1}$")
    if idx == 0:
        ax.legend()

# Clean up unused subplots if fewer than 6
for ax in axes[len(random_segments):]:
    fig.delaxes(ax)

plt.tight_layout()
plt.suptitle("Local vs Global Fit on 6 Random Segments (Dense Sliding Window)", fontsize=16, y=1.02)
plt.show()
