import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Setup parameters
runs = [1, 2, 3, 4, 5, 6, 8, 9]
n_rows, n_cols = 4, 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12), sharex=False, sharey=False)
axes = axes.flatten()
Q, R = 0.01, 0.01

# Load global model
with open("model_params.json", "r") as f:
    global_model = json.load(f)
A_g, B1_g, B2_g = global_model["A"], global_model["B1"], global_model["B2"]

# Load GMM model
with open("gmm_models_18.json", "r") as f:
    models = json.load(f)
Pi = np.ones((18, 18)) * 0.1
np.fill_diagonal(Pi, 0.6)
Pi /= Pi.sum(axis=1, keepdims=True)

# IMM Class Definition
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
        states = [omega_meas[0]] * self.num_models
        variances = [1.0] * self.num_models
        for k in range(1, N):
            likelihoods = []
            mixed_states, mixed_vars = [], []
            for i in range(self.num_models):
                x_mix = sum(self.Pi[i, j] * mu[j] * states[j] for j in range(self.num_models))
                P_mix = sum(self.Pi[i, j] * mu[j] * (variances[j] + (states[j] - x_mix)**2)
                            for j in range(self.num_models))
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
                likelihoods.append(np.exp(-0.5 * y**2 / S) / np.sqrt(2 * np.pi * S))

            mu = np.array([
                sum(self.Pi[j, i] * mu[j] * likelihoods[i] for j in range(self.num_models))
                for i in range(self.num_models)
            ])
            mu /= np.sum(mu)
            omega_est[k] = sum(mu[i] * states[i] for i in range(self.num_models))
        return omega_est

# Plot all runs
for ax, i in zip(axes, runs):
    base_path = Path("/media/asalvi/EEAA0057AA001EA9/ROSBAGS/ExtSkidB/IMMIceCSV")
    data_op = pd.read_csv(base_path / f'TSyn_0{i}.csv')
    data_ip = pd.read_csv(base_path / f'joint_states_0{i}.csv')
    omega = data_op["IMU_fil_AngZ"].values
    phiDotL = data_ip["velocity_0"].values
    phiDotR = data_ip["velocity_1"].values
    omega_meas = omega + np.random.uniform(0, 0.1, len(omega))
    time = np.arange(len(omega))

    # Global KF
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

    # IMM
    imm = IMMLinearOmega(models, Q, R, Pi)
    omega_imm = imm.run(omega_meas, phiDotL, phiDotR)

    ax.plot(time, omega, 'k-', label="True")
    ax.plot(time, omega_meas, 'orange', linestyle='--', label="Measurement")
    ax.plot(time, omega_kf, 'b:', label="Global KF")
    ax.plot(time, omega_imm, 'g-.', label="IMM")
    ax.set_title(f"Run {i}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Angular Velocity")
    ax.grid(True)
    ax.set_xlim([0, len(time)])
    y_min = min(np.min(omega), np.min(omega_kf), np.min(omega_imm)) - 0.2
    y_max = max(np.max(omega), np.max(omega_kf), np.max(omega_imm)) + 0.2
    ax.set_ylim([y_min, y_max])

axes[0].legend(loc='upper right')
plt.tight_layout()
plt.savefig("/home/asalvi/IMMIceResults/png/IMM_state_estimation_static_grid.png", dpi=300)
plt.show()
