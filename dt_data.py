import numpy as np

dts = np.array([
    # 0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
])

max_rwds = np.array([
    # 176.0,
    177.5,
    88.7,
    17.7,
    8.81,
    4.31,
    2.87,
    2.15,
    1.62,
    1.38,
    1.16,
    1.02,
    0.879,
    0.815,
])

wall_time = 1/dts
# Plot the data
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")
plt.figure(figsize=(10, 6))
# sns.pointplot(x=wall_time, y=max_rwds*dts)
plt.plot(wall_time, np.exp(-max_rwds*dts)*1000, marker='o', color='k')
# Draw a yellow dot at dt = 0.01
# plt.plot(1/0.01, np.exp(-88.7*0.01)*1000, marker='*', color='y', markersize=28)

plt.title("Effect of Simulation Timestep on Best Model\'s Performance")
plt.xlabel(f"Simulation Time ($1/\Delta$)")
plt.ylabel("Final Number of Cells")
plt.xscale('log')
# plt.yscale('log')
plt.savefig("dt_data.png", dpi=300, bbox_inches='tight')
plt.show()