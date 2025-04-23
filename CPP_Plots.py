import matplotlib.pyplot as plt
import pandas as pd

file_path = "./EmissionsVsTaxData_Chi0.10_SEED42_MODEAVERAGE.txt"
data = pd.read_csv(file_path, delim_whitespace=True, skiprows=1, names=["Emissions", "Tax", "Price of Goods"])

data["Time"] = range(1, len(data) + 1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

color_em = "red"
color_tax = "blue"
ax1.set_title("Emissions and Carbon Tax Over Time")
ax1.set_ylabel("Total Emissions", color=color_em)
ax1.plot(data["Time"], data["Emissions"], color=color_em, label="Emissions")
ax1.tick_params(axis='y', labelcolor=color_em)

ax1b = ax1.twinx()
ax1b.set_ylabel("Carbon Tax Level", color=color_tax)
ax1b.plot(data["Time"], data["Tax"], color=color_tax, linestyle="--", label="Carbon Tax")
ax1b.tick_params(axis='y', labelcolor=color_tax)

color_pg = "green"
ax2.set_title("Consumer Impact and Carbon Tax Over Time")
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Consumer Impact", color=color_pg)
ax2.plot(data["Time"], data["Price of Goods"], color=color_pg, label="Consumer Impact")
ax2.tick_params(axis='y', labelcolor=color_pg)

ax2b = ax2.twinx()
ax2b.set_ylabel("Carbon Tax Level", color=color_tax)
ax2b.plot(data["Time"], data["Tax"], color=color_tax, linestyle="--", label="Carbon Tax")
ax2b.tick_params(axis='y', labelcolor=color_tax)

plt.tight_layout()
plt.show()
