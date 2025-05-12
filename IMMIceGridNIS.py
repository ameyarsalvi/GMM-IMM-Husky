import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

from scipy.stats import chi2

# Chi-squared bounds for 1 DoF
chi2_lower = chi2.ppf(0.025, df=1)  # ≈ 0.00098
chi2_upper = chi2.ppf(0.975, df=1)  # ≈ 5.0239

SF = 2

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



# Define extended IMM class that returns both NIS and model probabilities
class IMMLinearOmegaWithNIS:
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
        S_all = np.zeros(N)
        innovations = np.zeros(N)

        for k in range(1, N):
            likelihoods = []
            mixed_states, mixed_vars = [], []
            for i in range(self.num_models):
                x_mix = sum(self.Pi[i, j] * mu[j] * states[j] for j in range(self.num_models))
                P_mix = sum(self.Pi[i, j] * mu[j] * (variances[j] + (states[j] - x_mix)**2)
                            for j in range(self.num_models))
                mixed_states.append(x_mix)
                mixed_vars.append(P_mix)

            x_preds, P_preds = [], []
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
                x_preds.append(x_pred)
                P_preds.append(P_pred)

            mu = np.array([sum(self.Pi[j, i] * mu[j] * likelihoods[i] for j in range(self.num_models))
                           for i in range(self.num_models)])
            mu /= np.sum(mu)
            omega_est[k] = sum(mu[i] * states[i] for i in range(self.num_models))
            model_probs[:, k] = mu
            combined_S = sum(mu[i] * (P_preds[i] + self.R) for i in range(self.num_models))
            combined_pred = sum(mu[i] * x_preds[i] for i in range(self.num_models))
            innovations[k] = omega_meas[k] - combined_pred
            S_all[k] = combined_S

        nis = (innovations ** 2) / (S_all + 1e-6)
        return nis, model_probs

# Load dataset and model
i = 9
base_path = Path("/media/asalvi/EEAA0057AA001EA9/ROSBAGS/ExtSkidB/IMMIceCSV")
data_op = pd.read_csv(base_path / f'TSyn_0{i}.csv')
data_ip = pd.read_csv(base_path / f'joint_states_0{i}.csv')
omega = data_op["IMU_fil_AngZ"].values
phiDotL = data_ip["velocity_0"].values
phiDotR = data_ip["velocity_1"].values
omega_meas = omega + np.random.uniform(0, 0.1, len(omega))
time = np.arange(len(omega))

with open("model_params.json", "r") as f:
    global_model = json.load(f)
A_g, B1_g, B2_g = global_model["A"], global_model["B1"], global_model["B2"]
Q, R = 0.01, 0.01

# Run global Kalman filter
N = len(omega_meas)
omega_kf = np.zeros(N)
P_kf = np.zeros(N)
omega_kf[0] = omega_meas[0]
P_kf[0] = 1.0
for k in range(1, N):
    u1, u2 = phiDotL[k-1], phiDotR[k-1]
    omega_pred = A_g * omega_kf[k-1] + B1_g * u1 + B2_g * u2
    P_pred = A_g**2 * P_kf[k-1] + Q
    y = omega_meas[k] - omega_pred
    S = P_pred + R
    K = P_pred / S
    omega_kf[k] = omega_pred + K * y
    P_kf[k] = (1 - K) * P_pred

# Plot NIS and IMM weights
gmm_components = [3, 6, 9, 18, 25]
fig, axes = plt.subplots(2, len(gmm_components), figsize=(8 * len(gmm_components), 10), sharex=True)

for idx, n_comp in enumerate(gmm_components):
    with open(f"gmm_models_{n_comp}.json", "r") as f:
        models = json.load(f)

    Pi = np.ones((n_comp, n_comp)) * 0.1
    np.fill_diagonal(Pi, 0.6)
    Pi /= Pi.sum(axis=1, keepdims=True)

    imm = IMMLinearOmegaWithNIS(models, Q, R, Pi)
    nis_imm, model_probs = imm.run(omega_meas, phiDotL, phiDotR)

    innovation_global = omega_meas - omega_kf
    S_global = P_kf + R
    nis_global = (innovation_global ** 2) / (S_global + 1e-6)

    # Top row: NIS
    ax1 = axes[0, idx]
    ax1.plot(time, nis_imm, label="IMM", color="green")
    ax1.plot(time, nis_global, label="KF", color="blue", linestyle="--")
    #ax1.axhline(3.84, linestyle=":", color="red", label="95% Bound" if idx == 0 else "")
    # Shaded 95% confidence interval band
    ax1.axhline(chi2_lower, linestyle=":", color="red", alpha=0.7, label="2.5% Bound" if idx == 0 else "")
    ax1.axhline(chi2_upper, linestyle=":", color="red", alpha=0.7, label="97.5% Bound" if idx == 0 else "")
    ax1.fill_between(time, chi2_lower, chi2_upper, color="gray", alpha=0.1, label="95% Interval" if idx == 0 else "")

    ax1.set_title(f"GMM-{n_comp} Components")
    if idx == 0:
        ax1.set_ylabel("NIS")
        #ax1.legend()
    ax1.grid(True)

    # Bottom row: IMM weights
    ax2 = axes[1, idx]
    for m in range(model_probs.shape[0]):
        ax2.plot(time, model_probs[m])
    if idx == 0:
        ax2.set_ylabel("Model Prob.")
    ax2.set_xlabel("Timestep")
    ax2.grid(True)

#plt.suptitle("Top: NIS Statistics | Bottom: IMM Weights (Per GMM Component Count)", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.tight_layout()

# Add shared legend above plots
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.04))


plt.savefig('/home/asalvi/IMMIceResults/eps/IMMIceNIS.eps', format='eps')

plt.savefig("/home/asalvi/IMMIceResults/png/IMMIceNIS.png", dpi=300)
