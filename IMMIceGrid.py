import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- IMM Class Definition ---
class IMMLinearOmega:
    def __init__(self, models, Q, R, Pi):
        self.models = models
        self.num_models = len(models)
        self.Q = Q
        self.R = R
        self.Pi = Pi

    def run(self, omega_meas, uL, uR):
        N = len(omega_meas)
        mu = np.ones(self.num_models) / self.num_models
        omega_est = np.zeros(N)
        model_probs = np.zeros((self.num_models, N))
        states = [omega_meas[0]] * self.num_models
        variances = [1.0] * self.num_models

        for k in range(1, N):
            likelihoods = []
            mixed_states, mixed_vars = [], []
            for i in range(self.num_models):
                x_mix = sum(self.Pi[i, j] * mu[j] * states[j] for j in range(self.num_models))
                P_mix = sum(
                    self.Pi[i, j] * mu[j] * (
                        variances[j] + (states[j] - x_mix)**2
                    ) for j in range(self.num_models)
                )
                mixed_states.append(x_mix)
                mixed_vars.append(P_mix)

            for i, model in enumerate(self.models):
                A, B1, B2 = model['A'], model['B1'], model['B2']
                u1, u2 = uL[k-1], uR[k-1]
                x_pred = A * mixed_states[i] + B1 * u1 + B2 * u2
                P_pred = A**2 * mixed_vars[i] + self.Q
                y = omega_meas[k] - x_pred
                S = P_pred + self.R
                K = P_pred / S
                x_upd = x_pred + K * y
                P_upd = (1 - K) * P_pred
                states[i] = x_upd
                variances[i] = P_upd
                likelihood = np.exp(-0.5 * y**2 / S) / np.sqrt(2 * np.pi * S)
                likelihoods.append(likelihood)

            mu = np.array([
                sum(self.Pi[j, i] * mu[j] * likelihoods[i] for j in range(self.num_models))
                for i in range(self.num_models)
            ])
            mu /= np.sum(mu)
            omega_est[k] = sum(mu[i] * states[i] for i in range(self.num_models))
            model_probs[:, k] = mu

        return omega_est, model_probs

# --- Load Data from Real Dataset ---
omega, phiDotL, phiDotR = [], [], []
i = 9

# Adjust path if needed
base_path = Path("/media/asalvi/EEAA0057AA001EA9/ROSBAGS/ExtSkidB/IMMIceCSV")
data_op = pd.read_csv(base_path / f'TSyn_0{i}.csv')
omega = data_op["IMU_fil_AngZ"].values

data_ip = pd.read_csv(base_path / f'joint_states_0{i}.csv')
phiDotL = data_ip["velocity_0"].values
phiDotR = data_ip["velocity_1"].values

omega_meas = omega + np.random.uniform(0, 0.1, len(omega))
time = np.arange(len(omega))

# --- Global Model ---
with open("model_params.json", "r") as f:
    global_model = json.load(f)
A_g, B1_g, B2_g = global_model["A"], global_model["B1"], global_model["B2"]
Q, R = 0.01, 0.01

# --- Global KF ---
omega_kf = np.zeros_like(omega)
P_kf = np.zeros_like(omega)
omega_kf[0] = omega_meas[0]
P_kf[0] = 1.0
for k in range(1, len(omega)):
    u1, u2 = phiDotL[k-1], phiDotR[k-1]
    omega_pred = A_g * omega_kf[k-1] + B1_g * u1 + B2_g * u2
    P_pred = A_g**2 * P_kf[k-1] + Q
    y = omega_meas[k] - omega_pred
    S = P_pred + R
    K = P_pred / S
    omega_kf[k] = omega_pred + K * y
    P_kf[k] = (1 - K) * P_pred

# --- 2x4 Grid of Estimations and Probabilities ---
fig, axes = plt.subplots(2, 4, figsize=(24, 8))
gmm_components = [3, 9, 18, 25]
zoom_start, zoom_end = 150, 200

for idx, n_comp in enumerate(gmm_components):
    with open(f"gmm_models_{n_comp}.json", "r") as f:
        models = json.load(f)

    Pi = np.ones((n_comp, n_comp)) * 0.1
    np.fill_diagonal(Pi, 0.6)
    Pi /= Pi.sum(axis=1, keepdims=True)

    imm = IMMLinearOmega(models, Q, R, Pi)
    omega_est, model_probs = imm.run(omega_meas, phiDotL, phiDotR)

    # Estimation plot (Top row)
    ax1 = axes[0, idx]
    ax1.plot(time, omega, label="True", color='black')
    ax1.plot(time, omega_meas, '--', label="Measurement", color='orange', alpha=0.6)
    ax1.plot(time, omega_est, '-.', label="IMM", color='green')
    ax1.plot(time, omega_kf, ':', label="KF", color='blue')
    ax1.set_title(f"GMM-{n_comp} Components")
    ax1.legend()
    ax1.grid(False)

    # Inset zoom
    axins = inset_axes(ax1, width="45%", height="55%", loc='lower right', borderpad=2)
    axins.plot(time, omega, color='black')
    axins.plot(time, omega_meas, '--', color='orange', alpha=0.6)
    axins.plot(time, omega_est, '-.', color='green')
    axins.plot(time, omega_kf, ':', color='blue')
    axins.set_xlim(zoom_start, zoom_end)
    axins.set_ylim(min(omega[zoom_start:zoom_end]) - 0.1, max(omega[zoom_start:zoom_end]) + 0.1)
    axins.set_xticks([])
    axins.set_yticks([])
    ax1.indicate_inset_zoom(axins, edgecolor="gray")

    # Model probabilities (Bottom row)
    ax2 = axes[1, idx]
    for m in range(model_probs.shape[0]):
        ax2.plot(model_probs[m], linewidth=1.0)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Model Prob." if idx == 0 else "")
    ax2.grid(False)

fig.suptitle("Top: State Estimation with Zoom | Bottom: IMM Model Probabilities", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()