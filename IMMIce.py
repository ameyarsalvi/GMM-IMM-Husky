import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# IMM class definition
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
            mixed_states = []
            mixed_vars = []

            # Mixing
            for i in range(self.num_models):
                x_mix = sum(self.Pi[i, j] * mu[j] * states[j] for j in range(self.num_models))
                P_mix = sum(
                    self.Pi[i, j] * mu[j] * (
                        variances[j] + (states[j] - x_mix)**2
                    ) for j in range(self.num_models)
                )
                mixed_states.append(x_mix)
                mixed_vars.append(P_mix)

            # Prediction + Update per model
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


# --- Load GMM models ---
with open("gmm_models_3.json", "r") as f:
    models = json.load(f)

# --- Load test data ---
omega, phiDotL, phiDotR = [], [], []
i = 9
data_op = pd.read_csv(f'/media/asalvi/EEAA0057AA001EA9/ROSBAGS/ExtSkidB/IMMIceCSV/TSyn_0{i}.csv')
omega.extend(data_op["IMU_fil_AngZ"].tolist())
data_ip = pd.read_csv(f'/media/asalvi/EEAA0057AA001EA9/ROSBAGS/ExtSkidB/IMMIceCSV/joint_states_0{i}.csv')
phiDotL.extend(data_ip["velocity_0"].tolist())
phiDotR.extend(data_ip["velocity_1"].tolist())

omega = np.array(omega)
phiDotL = np.array(phiDotL)
phiDotR = np.array(phiDotR)
omega_meas = omega + np.random.uniform(0, 0.1, len(omega))

# --- IMM parameters ---
Q = 0.01
R = 0.01
Pi = np.ones((len(models), len(models))) * 0.1
np.fill_diagonal(Pi, 0.6)
Pi /= Pi.sum(axis=1, keepdims=True)

# --- Run IMM ---
imm = IMMLinearOmega(models, Q, R, Pi)
omega_est, model_probs = imm.run(omega_meas, phiDotL, phiDotR)

# --- Load global model ---
with open("model_params.json", "r") as f:
    global_model = json.load(f)
A_g, B1_g, B2_g = global_model["A"], global_model["B1"], global_model["B2"]

# --- Global Kalman Filter ---
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

# --- Time vector ---
time = np.arange(len(omega))

# --- Plot: IMM, KF, True, Measurement with Inset Zoom ---
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(time, omega, label="True Angular Velocity", linewidth=2, color='black')
ax.plot(time, omega_meas, label="Noisy Measurement", linestyle='--', color='orange', alpha=0.6)
ax.plot(time, omega_est, label="IMM Estimated", linestyle='-.', color='green', linewidth=2)
ax.plot(time, omega_kf, label="Global KF Estimated", linestyle=':', color='blue', linewidth=2)

ax.set_xlabel("Timestep")
ax.set_ylabel("Angular Velocity")
ax.set_title("IMM vs Global Kalman Filter: State Estimation")
ax.legend()
ax.grid(False)

# --- Inset Zoom: x = 120 to 220 ---
axins = inset_axes(ax, width="45%", height="55%", loc='lower right', borderpad=2)
zoom_start = 150
zoom_end = 200

axins.plot(time, omega, color='black')
axins.plot(time, omega_meas, linestyle='--', color='orange', alpha=0.6)
axins.plot(time, omega_est, linestyle='-.', color='green')
axins.plot(time, omega_kf, linestyle=':', color='blue')
axins.set_xlim(zoom_start, zoom_end)
axins.set_ylim(
    min(omega[zoom_start:zoom_end]) - 0.1,
    max(omega[zoom_start:zoom_end]) + 0.1
)
axins.grid(True)
axins.set_xticks([])
axins.set_yticks([])
ax.indicate_inset_zoom(axins, edgecolor="gray")

plt.tight_layout()
plt.show()

# --- RMSEs ---
rmse_imm = np.sqrt(np.mean((omega_meas - omega_est)**2))
rmse_kf = np.sqrt(np.mean((omega_meas - omega_kf)**2))
print(f"IMM Estimation RMSE:       {rmse_imm:.5f}")
print(f"Global KF Estimation RMSE: {rmse_kf:.5f}")

# --- Plot Model Probabilities ---
plt.figure(figsize=(14, 6))
for i in range(model_probs.shape[0]):
    plt.plot(model_probs[i], label=f'Model {i}', linewidth=1.5)
plt.xlabel("Timestep")
plt.ylabel("Model Probability")
plt.title("IMM Model Probabilities Over Time")
plt.legend(loc='upper right', ncol=3)
plt.grid(False)
plt.tight_layout()
plt.show()
