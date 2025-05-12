import numpy as np
import json
from pathlib import Path
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SF = 1.0

SMALL_SIZE = 16*SF
MEDIUM_SIZE = 18*SF
BIGGER_SIZE = 20*SF

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# ---- Step 1: Load all segment models ----
segment_dir = Path("local_model_segments_dense")
model_vecs = []

for path in sorted(segment_dir.glob("segment_*.json")):
    with open(path, 'r') as f:
        data = json.load(f)
        model_vecs.append([data['A'], data['B1'], data['B2']])

model_vecs = np.array(model_vecs)  # shape: (61, 3)

# ---- Step 2: Fit GMM with 5 components ----
n_components = 15
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
gmm.fit(model_vecs)

means = gmm.means_           # shape (5, 3)
covs = gmm.covariances_      # shape (5, 3, 3)
weights = gmm.weights_       # shape (5,)
labels = gmm.predict(model_vecs)  # cluster assignment for each model

# ---- Step 3: Save the GMM cluster means as representative models ----
gmm_representatives = []
for i, mean in enumerate(means):
    rep = {
        "cluster": i,
        "A": float(mean[0]),
        "B1": float(mean[1]),
        "B2": float(mean[2]),
        "weight": float(weights[i])
    }
    gmm_representatives.append(rep)

with open("gmm_models_15.json", "w") as f:
    json.dump(gmm_representatives, f, indent=4)

print(f"Saved {n_components} representative models to 'gmm_models_15.json'.")

# ---- Step 4: Visualize the clustering in 3D ----
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each local model with color-coded cluster
ax.scatter(model_vecs[:, 0], model_vecs[:, 1], model_vecs[:, 2],
           c=labels, cmap='viridis', label='Local Models', s=50, alpha=0.8)

# Plot cluster means
ax.scatter(means[:, 0], means[:, 1], means[:, 2],
           color='red', s=150, marker='X', label='GMM Cluster Means')

ax.set_xlabel("A")
ax.set_ylabel("B1")
ax.set_zlabel("B2")
ax.set_title("GMM Clustering of Local Models in Parameter Space")
ax.legend()
plt.tight_layout()
plt.show()

# ---- Step 5: Plot 2D projections in a 1x3 grid ----
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))

# A vs B1
axes[0].scatter(model_vecs[:, 0], model_vecs[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
axes[0].scatter(means[:, 0], means[:, 1], color='red', s=150, marker='X')
axes[0].set_xlabel("A")
axes[0].set_ylabel("B1")
axes[0].set_title("A vs B1")

# A vs B2
axes[1].scatter(model_vecs[:, 0], model_vecs[:, 2], c=labels, cmap='viridis', s=50, alpha=0.7)
axes[1].scatter(means[:, 0], means[:, 2], color='red', s=150, marker='X')
axes[1].set_xlabel("A")
axes[1].set_ylabel("B2")
axes[1].set_title("A vs B2")

# B1 vs B2
axes[2].scatter(model_vecs[:, 1], model_vecs[:, 2], c=labels, cmap='viridis', s=50, alpha=0.7)
axes[2].scatter(means[:, 1], means[:, 2], color='red', s=150, marker='X')
axes[2].set_xlabel("B1")
axes[2].set_ylabel("B2")
axes[2].set_title("B1 vs B2")

for ax in axes:
    ax.grid(True)

plt.suptitle("2D Projections of GMM Clustering", fontsize=16)
plt.tight_layout()
plt.show()
