import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Define paths and settings
run_ids = [f"{i:02d}" for i in range(1, 10)]
gmm_components = [3, 5, 6, 9, 10, 12, 15, 18, 20, 25]
data_path = Path("/media/asalvi/EEAA0057AA001EA9/ROSBAGS/ExtSkidB/IMMIceCSV/")
global_model_path = Path("model_params.json")

# Load global model
with open(global_model_path, "r") as f:
    global_model = json.load(f)
A_g, B1_g, B2_g = global_model["A"], global_model["B1"], global_model["B2"]

Q = 0.01
R = 0.01

# Kalman Filter implementation for global model
def run_global_kf(omega_meas, phiDotL, phiDotR):
    omega_kf = np.zeros_like(omega_meas)
    P_kf = np.zeros_like(omega_meas)
    omega_kf[0] = omega_meas[0]
    P_kf[0] = 1.0
    for k in range(1, len(omega_meas)):
        u1, u2 = phiDotL[k-1], phiDotR[k-1]
        omega_pred = A_g * omega_kf[k-1] + B1_g * u1 + B2_g * u2
        P_pred = A_g**2 * P_kf[k-1] + Q
        y = omega_meas[k] - omega_pred
        S = P_pred + R
        K = P_pred / S
        omega_kf[k] = omega_pred + K * y
        P_kf[k] = (1 - K) * P_pred
    return omega_kf

# IMM class
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
            mixed_states = []
            mixed_vars = []
            for i in range(self.num_models):
                x_mix = sum(self.Pi[i, j] * mu[j] * states[j] for j in range(self.num_models))
                P_mix = sum(self.Pi[i, j] * mu[j] * (variances[j] + (states[j] - x_mix)**2)
                            for j in range(self.num_models))
                mixed_states.append(x_mix)
                mixed_vars.append(P_mix)
            likelihoods = []
            for i, model in enumerate(self.models):
                A, B1, B2 = model["A"], model["B1"], model["B2"]
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

# Containers for RMSEs
rmse_results = {"global": []}
for n_comp in gmm_components:
    rmse_results[f"gmm_{n_comp}"] = []

# Sweep through runs
for run in run_ids:
    # Load data
    data_op = pd.read_csv(data_path / f"TSyn_{run}.csv")
    data_ip = pd.read_csv(data_path / f"joint_states_{run}.csv")
    omega = np.array(data_op["IMU_fil_AngZ"])
    phiDotL = np.array(data_ip["velocity_0"])
    phiDotR = np.array(data_ip["velocity_1"])
    omega_meas = omega + np.random.uniform(0, 0.1, len(omega))

    # Global model
    omega_kf = run_global_kf(omega_meas, phiDotL, phiDotR)
    rmse_results["global"].append(float(np.sqrt(np.mean((omega_meas - omega_kf) ** 2))))

    # GMM models
    for n_comp in gmm_components:
        with open(f"gmm_models_{n_comp}.json", "r") as f:
            models = json.load(f)
        Pi = np.ones((n_comp, n_comp)) * 0.1
        np.fill_diagonal(Pi, 0.6)
        Pi /= Pi.sum(axis=1, keepdims=True)
        imm = IMMLinearOmega(models, Q, R, Pi)
        omega_est, _ = imm.run(omega_meas, phiDotL, phiDotR)
        rmse = np.sqrt(np.mean((omega_meas - omega_est) ** 2))
        rmse_results[f"gmm_{n_comp}"].append(float(rmse))

# Save results to JSON
with open("rmse_summary.json", "w") as f:
    json.dump(rmse_results, f, indent=4)

# Compute mean RMSEs
mean_rmse = {k: np.mean(v) for k, v in rmse_results.items()}

# Bar Plot
labels = ["global"] + [f"gmm_{n}" for n in gmm_components]
values = [mean_rmse[k] for k in labels]

plt.figure(figsize=(12, 6))
bars = plt.bar(labels, values, color='skyblue')
plt.ylabel("Mean RMSE")
plt.title("Mean RMSE: Global vs GMM-IMM Models (Over 9 Runs)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True, axis='y')
plt.show()
