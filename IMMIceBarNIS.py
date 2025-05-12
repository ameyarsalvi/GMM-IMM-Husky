import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.stats import chi2

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# Define chi-squared bounds
lower_bound = chi2.ppf(0.025, df=1)
upper_bound = chi2.ppf(0.975, df=1)

# Setup
run_ids_main = [f"{i:02d}" for i in range(1, 10)]
run_ids_extra = [f"{i:02d}" for i in range(10, 12)]
gmm_components = [3, 5, 6, 9, 10, 12, 15, 18, 20, 25]
data_path = Path("/media/asalvi/EEAA0057AA001EA9/ROSBAGS/ExtSkidB/IMMIceCSV/")
global_model_path = Path("model_params.json")
Q, R = 0.01, 0.01

# Load global model
with open(global_model_path, "r") as f:
    global_model = json.load(f)
A_g, B1_g, B2_g = global_model["A"], global_model["B1"], global_model["B2"]

# Global KF
def run_global_kf_with_nis(omega_meas, phiDotL, phiDotR):
    N = len(omega_meas)
    omega_kf = np.zeros(N)
    P_kf = np.zeros(N)
    omega_kf[0] = omega_meas[0]
    P_kf[0] = 1.0
    innovations = np.zeros(N)
    S_all = np.zeros(N)
    for k in range(1, N):
        u1, u2 = phiDotL[k-1], phiDotR[k-1]
        omega_pred = A_g * omega_kf[k-1] + B1_g * u1 + B2_g * u2
        P_pred = A_g**2 * P_kf[k-1] + Q
        y = omega_meas[k] - omega_pred
        S = P_pred + R
        K = P_pred / S
        omega_kf[k] = omega_pred + K * y
        P_kf[k] = (1 - K) * P_pred
        innovations[k] = y
        S_all[k] = S
    nis = (innovations ** 2) / (S_all + 1e-6)
    return nis

# IMM
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
        states = [omega_meas[0]] * self.num_models
        variances = [1.0] * self.num_models
        innovations = np.zeros(N)
        S_all = np.zeros(N)
        for k in range(1, N):
            mixed_states, mixed_vars = [], []
            for i in range(self.num_models):
                x_mix = sum(self.Pi[i, j] * mu[j] * states[j] for j in range(self.num_models))
                P_mix = sum(self.Pi[i, j] * mu[j] * (variances[j] + (states[j] - x_mix)**2)
                            for j in range(self.num_models))
                mixed_states.append(x_mix)
                mixed_vars.append(P_mix)

            x_preds, P_preds, likelihoods = [], [], []
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
                x_preds.append(x_pred)
                P_preds.append(P_pred)
                likelihoods.append(np.exp(-0.5 * y**2 / S) / np.sqrt(2 * np.pi * S))

            mu = np.array([sum(self.Pi[j, i] * mu[j] * likelihoods[i] for j in range(self.num_models))
                           for i in range(self.num_models)])
            mu /= np.sum(mu)
            combined_S = sum(mu[i] * (P_preds[i] + self.R) for i in range(self.num_models))
            combined_pred = sum(mu[i] * x_preds[i] for i in range(self.num_models))
            innovations[k] = omega_meas[k] - combined_pred
            S_all[k] = combined_S

        nis = (innovations ** 2) / (S_all + 1e-6)
        return nis

# Helper to compute stats
def compute_nis_stats(run_ids):
    over_conf_stats = {"global": []}
    under_conf_stats = {"global": []}
    for n in gmm_components:
        over_conf_stats[f"gmm_{n}"] = []
        under_conf_stats[f"gmm_{n}"] = []

    for run in run_ids:
        data_op = pd.read_csv(data_path / f"TSyn_{run}.csv")
        data_ip = pd.read_csv(data_path / f"joint_states_{run}.csv")
        omega = np.array(data_op["IMU_fil_AngZ"])
        phiDotL = np.array(data_ip["velocity_0"])
        phiDotR = np.array(data_ip["velocity_1"])
        omega_meas = omega + np.random.uniform(0, 0.1, len(omega))

        # Global
        nis_global = run_global_kf_with_nis(omega_meas, phiDotL, phiDotR)
        over_conf_stats["global"].append(np.mean(nis_global > upper_bound))
        under_conf_stats["global"].append(np.mean(nis_global < lower_bound))

        # GMM models
        for n_comp in gmm_components:
            with open(f"gmm_models_{n_comp}.json", "r") as f:
                models = json.load(f)
            Pi = np.ones((n_comp, n_comp)) * 0.1
            np.fill_diagonal(Pi, 0.6)
            Pi /= Pi.sum(axis=1, keepdims=True)
            imm = IMMLinearOmegaWithNIS(models, Q, R, Pi)
            nis_imm = imm.run(omega_meas, phiDotL, phiDotR)
            over_conf_stats[f"gmm_{n_comp}"].append(np.mean(nis_imm > upper_bound))
            under_conf_stats[f"gmm_{n_comp}"].append(np.mean(nis_imm < lower_bound))

    return over_conf_stats, under_conf_stats

# Compute both sets
over_main, under_main = compute_nis_stats(run_ids_main)
over_extra, under_extra = compute_nis_stats(run_ids_extra)

# Bar chart setup
labels = ["global"] + [f"gmm_{n}" for n in gmm_components]
Xlabels = ["global", "3", "5","6", "9","10","12","15","18", "20" ,"25"]
x = np.arange(len(labels))
bar_width = 0.4

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Over-confidence plot
main_vals = [np.mean(over_main[k]) for k in labels]
extra_vals = [np.mean(over_extra[k]) for k in labels]
ax1.bar(x - bar_width/2, main_vals, width=bar_width, color="tomato", label="Dataset A")
ax1.bar(x + bar_width/2, extra_vals, width=bar_width, color="firebrick", label="Dataset B")
ax1.set_ylabel("Fraction NIS > 97.5%")
ax1.set_title("Over-Confidence: NIS > Upper Chi-Squared Bound")
ax1.legend()
ax1.grid(True, axis="y")

# Under-confidence plot
main_vals = [np.mean(under_main[k]) for k in labels]
extra_vals = [np.mean(under_extra[k]) for k in labels]
ax2.bar(x - bar_width/2, main_vals, width=bar_width, color="steelblue", label="Dataset A")
ax2.bar(x + bar_width/2, extra_vals, width=bar_width, color="midnightblue", label="Dataset B")
ax2.set_ylabel("Fraction NIS < 2.5%")
ax2.set_title("Under-Confidence: NIS < Lower Chi-Squared Bound")
ax2.set_xlabel("Model")
ax2.legend()
ax2.grid(True, axis="y")
plt.xticks(x, Xlabels)



plt.tight_layout()
#plt.show()

plt.savefig('/home/asalvi/IMMIceResults/eps/IMMIceBar.eps', format='eps')

plt.savefig("/home/asalvi/IMMIceResults/png/IMMIceBar.png", dpi=300)