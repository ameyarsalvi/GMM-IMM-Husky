import numpy as np
import pandas as pd
import json
from pathlib import Path

# --- Step 1: Load and stack all data ---
omega = []
phiDotL = []
phiDotR = []

for i in range(9):
    data_op = pd.read_csv(f'/media/asalvi/EEAA0057AA001EA9/ROSBAGS/ExtSkidB/IMMIceCSV/TSyn_0{i+1}.csv')
    omega.extend(data_op["IMU_fil_AngZ"].tolist())

    data_ip = pd.read_csv(f'/media/asalvi/EEAA0057AA001EA9/ROSBAGS/ExtSkidB/IMMIceCSV/joint_states_0{i+1}.csv')
    phiDotL.extend(data_ip["velocity_0"].tolist())
    phiDotR.extend(data_ip["velocity_1"].tolist())

omega = np.array(omega)
phiDotL = np.array(phiDotL)
phiDotR = np.array(phiDotR)

# --- Step 2: Moving horizon sequence extraction ---
sequence_length = 25
num_sequences = len(omega) - sequence_length

output_dir = Path("local_model_segments_dense")
output_dir.mkdir(exist_ok=True)

local_models = []

for i in range(num_sequences):
    idx_start = i
    idx_end = i + sequence_length

    x_k = omega[idx_start:idx_end-1]
    x_k_next = omega[idx_start+1:idx_end]
    u_k = np.vstack((phiDotL[idx_start:idx_end-1], phiDotR[idx_start:idx_end-1]))

    X = np.vstack((x_k, u_k))
    AB = x_k_next @ np.linalg.pinv(X)
    A_i, B1_i, B2_i = AB

    # Save model + data
    segment_data = {
        "segment_index": i,
        "A": float(A_i),
        "B1": float(B1_i),
        "B2": float(B2_i),
        "omega": x_k.tolist(),
        "omega_next": x_k_next.tolist(),
        "phiDotL": u_k[0].tolist(),
        "phiDotR": u_k[1].tolist()
    }

    with open(output_dir / f"segment_{i:04d}.json", "w") as f:
        json.dump(segment_data, f, indent=4)

    local_models.append(segment_data)

print(f"Saved {len(local_models)} dense local models to '{output_dir}'")
