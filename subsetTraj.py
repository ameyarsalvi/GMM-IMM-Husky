import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

SF = 1.0

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

# --- Load data ---
base_path = Path("/media/asalvi/EEAA0057AA001EA9/ROSBAGS/ExtSkidB/IMMIceCSV")
data_op = pd.read_csv(base_path / 'TSyn_01.csv')
omega = data_op["IMU_fil_AngZ"].values

# --- Plot base signal ---
plt.figure(figsize=(7, 7))     # A perfect square
plt.plot(omega[120:300], label="Omega", color='black')
plt.xlabel('Data Samples')
plt.ylabel('Omega [rad/s]')
#plt.title("Visualization of Moving Window Segments for Model Fitting")

# --- Parameters for overlay ---
segment_length = 25
start_indices = [0, 10, 20, 30]  # relative to offset
offset = 120
y_base = np.max(omega[120:300]) + 0.2
y_step = 0.25             # Increased spacing between lines
v_height = 0.03
text_offset = 0.08        # Text now goes ABOVE the line
font_size = 16

# --- Draw segments and labels ---
for i, s in enumerate(start_indices):
    x_start = s
    x_end = s + segment_length - 1
    y_level = y_base - i * y_step

    # Horizontal red line
    plt.hlines(y=y_level, xmin=x_start, xmax=x_end, colors='red', linewidth=2)

    # Vertical end ticks
    plt.vlines(x=x_start, ymin=y_level - v_height, ymax=y_level + v_height, colors='red', linewidth=2)
    plt.vlines(x=x_end, ymin=y_level - v_height, ymax=y_level + v_height, colors='red', linewidth=2)

    # Annotate text ABOVE the line
    mid_x = (x_start + x_end) / 2
    text = r"$A^{%d},\ B_1^{%d},\ B_2^{%d}$" % (i+1, i+1, i+1)
    plt.text(mid_x, y_level + text_offset, text, fontsize=font_size, color='blue', ha='center')

# --- Final touches ---
plt.ylim([np.min(omega[120:300]) - 0.2, y_base + 0.4])
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.show()

plt.savefig('/home/asalvi/IMMIceResults/eps/WindowVis.eps', format='eps')
plt.savefig("/home/asalvi/IMMIceResults/png/WindowVis.png", dpi=300)