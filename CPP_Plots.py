import matplotlib.pyplot as plt
import pandas as pd

file_path = "./EmissionsVsTaxData.txt"
data = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, names=["Emissions", "Tax"])

fig, ax1 = plt.subplots(figsize=(16, 6))

ax1.plot(data["Emissions"], color="red", label="Total Emissions")
ax1.set_ylabel("Total Emissions", color="red")
ax1.set_xlabel("Time Steps")

policy_start = 10
ax1.axvline(policy_start, color="black", linestyle="--", label="Policy Start")

ax2 = ax1.twinx()
ax2.plot(data["Tax"], color="blue", linestyle="--", label="Carbon Tax")
ax2.set_ylabel("Carbon Tax Level", color="blue")

plt.title("Emissions and Carbon Tax Over Time C++")
fig.legend(loc="upper right")

plt.tight_layout()
plt.show()