import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D


SF = 1

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

# Load model vectors
segment_dir = Path("local_model_segments_dense")
model_vecs = []

for path in sorted(segment_dir.glob("segment_*.json")):
    with open(path, 'r') as f:
        data = json.load(f)
        model_vecs.append([data['A'], data['B1'], data['B2']])

model_vecs = np.array(model_vecs)
component_list = [3, 6, 9, 12, 18, 25]

# ---- 1x6 3D Plot Grid ----
fig3d = plt.figure(figsize=(24, 7))
for idx, n_components in enumerate(component_list):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(model_vecs)
    means = gmm.means_
    labels = gmm.predict(model_vecs)
    weights = gmm.weights_

    # Save GMM models
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

    with open(f"gmm_models_{n_components}.json", "w") as f:
        json.dump(gmm_representatives, f, indent=4)

    # Plot in 3D grid
    ax = fig3d.add_subplot(1, 6, idx + 1, projection='3d')
    ax.scatter(model_vecs[:, 0], model_vecs[:, 1], model_vecs[:, 2],
               c=labels, cmap='viridis', s=30, alpha=0.7)
    ax.scatter(means[:, 0], means[:, 1], means[:, 2],
               color='red', s=200, marker='X', label='Means')
    ax.set_xlabel("A")
    ax.set_ylabel("B1")
    ax.set_zlabel("B2")
    ax.set_title(f"GMM ({n_components})")
    ax.view_init(elev=18, azim=120)
    ax.grid(True)

plt.suptitle("GMM Clustering (3D View)", fontsize=18, y=1.02)
plt.tight_layout()
#plt.show()

# ---- 3x6 Grid of 2D Projections ----
fig2d, axes = plt.subplots(3, 6, figsize=(26, 12))
for idx, n_components in enumerate(component_list):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(model_vecs)
    means = gmm.means_
    labels = gmm.predict(model_vecs)

    # A vs B1
    ax = axes[0, idx]
    ax.scatter(model_vecs[:, 0], model_vecs[:, 1], c=labels, cmap='viridis', s=30, alpha=0.7)
    ax.scatter(means[:, 0], means[:, 1], color='red', s=100, marker='X')
    if idx == 0: ax.set_ylabel("B1")
    ax.set_xlabel("A")
    ax.set_title(f"GMM ({n_components})")
    ax.grid(True)

    # A vs B2
    ax = axes[1, idx]
    ax.scatter(model_vecs[:, 0], model_vecs[:, 2], c=labels, cmap='viridis', s=30, alpha=0.7)
    ax.scatter(means[:, 0], means[:, 2], color='red', s=100, marker='X')
    if idx == 0: ax.set_ylabel("B2")
    ax.set_xlabel("A")
    ax.grid(True)

    # B1 vs B2
    ax = axes[2, idx]
    ax.scatter(model_vecs[:, 1], model_vecs[:, 2], c=labels, cmap='viridis', s=30, alpha=0.7)
    ax.scatter(means[:, 1], means[:, 2], color='red', s=100, marker='X')
    if idx == 0: ax.set_ylabel("B2")
    ax.set_xlabel("B1")
    ax.grid(True)

fig2d.suptitle("2D Projections of GMM Clustering", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.97])
#plt.show()

plt.savefig('/home/asalvi/IMMIceResults/eps/GMMModels.eps', format='eps')

plt.savefig("/home/asalvi/IMMIceResults/png/GMMModels.png", dpi=300)