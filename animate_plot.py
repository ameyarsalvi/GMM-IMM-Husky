# Re-import necessary libraries after kernel reset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from matplotlib.animation import FuncAnimation, PillowWriter

# Reload data and variables
i = 9
base_path = Path("/media/asalvi/EEAA0057AA001EA9/ROSBAGS/ExtSkidB/IMMIceCSV")
data_op = pd.read_csv(base_path / f'TSyn_0{i}.csv')
data_ip = pd.read_csv(base_path / f'joint_states_0{i}.csv')
omega = data_op["IMU_fil_AngZ"].values
phiDotL = data_ip["velocity_0"].values
phiDotR = data_ip["velocity_1"].values
omega_meas = omega + np.random.uniform(0, 0.1, len(omega))
time = np.arange(len(omega))
Q, R = 0.01, 0.01

with open("model_params.json", "r") as f:
    global_model = json.load(f)
A_g, B1_g, B2_g = global_model["A"], global_model["B1"], global_model["B2"]

omega_kf = np.zeros_like(omega)
P_kf = np.zeros_like(omega)
omega_kf[0] = omega_meas[0]
P_kf[0] = 1.0
for k in range(1, len(omega)):
    u1, u2 = phiDotL[k - 1], phiDotR[k - 1]
    omega_pred = A_g * omega_kf[k - 1] + B1_g * u1 + B2_g * u2
    P_pred = A_g**2 * P_kf[k - 1] + Q
    y = omega_meas[k] - omega_pred
    S = P_pred + R
    K = P_pred / S
    omega_kf[k] = omega_pred + K * y
    P_kf[k] = (1 - K) * P_pred

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
                P_mix = sum(self.Pi[i, j] * mu[j] * (
                        variances[j] + (states[j] - x_mix) ** 2) for j in range(self.num_models))
                mixed_states.append(x_mix)
                mixed_vars.append(P_mix)

            for i, model in enumerate(self.models):
                A, B1, B2 = model['A'], model['B1'], model['B2']
                u1, u2 = uL[k - 1], uR[k - 1]
                x_pred = A * mixed_states[i] + B1 * u1 + B2 * u2
                P_pred = A ** 2 * mixed_vars[i] + self.Q
                y = omega_meas[k] - x_pred
                S = P_pred + self.R
                K = P_pred / S
                x_upd = x_pred + K * y
                P_upd = (1 - K) * P_pred
                states[i] = x_upd
                variances[i] = P_upd
                likelihood = np.exp(-0.5 * y ** 2 / S) / np.sqrt(2 * np.pi * S)
                likelihoods.append(likelihood)

            mu = np.array([
                sum(self.Pi[j, i] * mu[j] * likelihoods[i] for j in range(self.num_models))
                for i in range(self.num_models)
            ])
            mu /= np.sum(mu)
            omega_est[k] = sum(mu[i] * states[i] for i in range(self.num_models))
            model_probs[:, k] = mu

        return omega_est, model_probs

with open("gmm_models_18.json", "r") as f:
    models = json.load(f)
Pi = np.ones((18, 18)) * 0.1
np.fill_diagonal(Pi, 0.6)
Pi /= Pi.sum(axis=1, keepdims=True)
imm = IMMLinearOmega(models, Q, R, Pi)
omega_imm, model_probs = imm.run(omega_meas, phiDotL, phiDotR)

# Plot + Animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.set_title("State Estimation")
ax2.set_title("IMM Model Probabilities")
ax1.set_xlim(0, len(time))
ax1.set_ylim(min(omega) - 0.2, max(omega) + 0.2)
ax2.set_xlim(0, len(time))
ax2.set_ylim(0, 1)

line_true, = ax1.plot([], [], 'k-', label="True")
line_meas, = ax1.plot([], [], 'orange', linestyle='--', label="Measurement")
line_kf, = ax1.plot([], [], 'b:', label="Global KF")
line_imm, = ax1.plot([], [], 'g-.', label="IMM")
model_lines = [ax2.plot([], [], label=f"m{i}")[0] for i in range(18)]
ax1.legend(loc='upper right')

def init():
    line_true.set_data([], [])
    line_meas.set_data([], [])
    line_kf.set_data([], [])
    line_imm.set_data([], [])
    for line in model_lines:
        line.set_data([], [])
    return [line_true, line_meas, line_kf, line_imm] + model_lines

def update(frame):
    x = time[:frame]
    line_true.set_data(x, omega[:frame])
    line_meas.set_data(x, omega_meas[:frame])
    line_kf.set_data(x, omega_kf[:frame])
    line_imm.set_data(x, omega_imm[:frame])
    for i, line in enumerate(model_lines):
        line.set_data(x, model_probs[i, :frame])
    return [line_true, line_meas, line_kf, line_imm] + model_lines

ani = FuncAnimation(fig, update, frames=len(time), init_func=init,
                    blit=True, interval=100, repeat=False)

ani_path = "/home/asalvi/IMMIceResults/IMM_GMM18_Run9.gif"
ani.save(ani_path, writer=PillowWriter(fps=10))
ani_path
