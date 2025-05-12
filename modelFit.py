import numpy as np
import pandas as pd

#vars = ['01', '02', '03', '04', '05', '06', '07', '08', '09']

omega = []
phiDotL = []
phiDotR = []


for i in range(9):

    data_op = pd.read_csv(f'/media/asalvi/EEAA0057AA001EA9/ROSBAGS/ExtSkidB/IMMIceCSV/TSyn_0{i+1}.csv')
    #print(len(data_op["IMU_fil_AngZ"]))
    omega.extend(data_op["IMU_fil_AngZ"].tolist())

    data_ip = pd.read_csv(f'/media/asalvi/EEAA0057AA001EA9/ROSBAGS/ExtSkidB/IMMIceCSV/joint_states_0{i+1}.csv')
    phiDotL.extend(data_ip["velocity_0"].tolist())
    phiDotR.extend(data_ip["velocity_1"].tolist())

print(f"shape of omega is{np.shape(omega)}")
print(f"shape of input is is{np.shape([phiDotL,phiDotR])}")


# Convert to numpy arrays
omega = np.array(omega)
phiDotL = np.array(phiDotL)
phiDotR = np.array(phiDotR)

# Build x_k and x_{k+1}
x_k = omega[:-1].reshape(1, -1)  # shape: (1, N-1)
x_next = omega[1:]              # shape: (N-1,)

# Build u_k
u_k = np.vstack((phiDotL[:-1], phiDotR[:-1]))  # shape: (2, N-1)

# Concatenate state and control
X = np.vstack((x_k, u_k))  # shape: (3, N-1)

# Solve for [A, B1, B2] using least squares
AB = x_next @ np.linalg.pinv(X)  # shape: (1, 3)

A = AB[0]
B1 = AB[1]
B2 = AB[2]

print(f"Estimated A: {A:.5f}")
print(f"Estimated B1 (phiDotL): {B1:.5f}")
print(f"Estimated B2 (phiDotR): {B2:.5f}")
    
import json

# Pack your model parameters
model_params = {
    'A': float(A),
    'B1': float(B1),
    'B2': float(B2)
}

# Save to JSON file
with open('model_params.json', 'w') as f:
    json.dump(model_params, f, indent=4)

print("Model parameters saved to model_params.json")
