import matplotlib.pyplot as plt
import pandas as pd

#plt.style.use('seaborn-v0_8-whitegrid')  # Consistent, clean look

# === LOAD DATA (as before) ===
file_path = "./EmissionsVsTaxData_Chi0.10_SEED42_MODEPESSIMISTIC.txt"
data = pd.read_csv(
    file_path,
    delim_whitespace=True,
    skiprows=1,
    names=["Emissions", "Tax", "Price of Goods"]
)
data["Time"] = range(1, len(data) + 1)

# === Calculate % decrease and % increase ===
baseline = data[data["Time"] <= 10]
base_em = baseline["Emissions"].mean()
base_pg = baseline["Price of Goods"].mean()
data_pct = data[data["Time"] > 10].copy()
data_pct["PctDecreaseEmissions"] = (
    (base_em - data_pct["Emissions"]) / base_em * 100
)
data_pct["PctIncreasePriceOfGoods"] = (
    (data_pct["Price of Goods"] - base_pg) / base_pg * 100
)

# === Add initial zero row for continuity ===
initial_row = pd.DataFrame({
    "Time": [0],
    "PctDecreaseEmissions": [0],
    "PctIncreasePriceOfGoods": [0],
    "Tax": [0]
})
data_pct_reset = pd.concat([initial_row, data_pct[["Time", "PctDecreaseEmissions", "PctIncreasePriceOfGoods", "Tax"]]], ignore_index=True)
data_pct_reset["Time"] = range(1, len(data_pct_reset) + 1)

# === Main Figure: Emissions vs Tax ===
fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(
    data_pct_reset["Time"], data_pct_reset["PctDecreaseEmissions"],
    color="red", linewidth=2, label="Emissions Decrease (%)"
)
ax1.set_xlabel("Time Step", fontsize=20)
ax1.set_ylabel("Decrease in Emissions (%)", fontsize=20, color="red")
ax1.tick_params(axis="y", labelcolor="red", labelsize=15)
ax1.set_ylim(0, 100)
ax1.set_xlim(0, data_pct_reset["Time"].max())
ax1.grid(True, which='major', axis='both', linestyle='--', linewidth=0.7)

# Carbon Tax axis
ax1b = ax1.twinx()
ax1b.plot(
    data_pct_reset["Time"], data_pct_reset["Tax"],
    color="blue", linestyle="--", linewidth=2, label="Carbon Tax"
)
ax1b.set_ylabel("Carbon Tax Level", fontsize=20, color="blue")
ax1b.tick_params(axis="y", labelcolor="blue", labelsize=15)
ax1b.set_ylim(0, 5.0)

# Legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=15)

#plt.title("Percentage Decrease in Emissions and Carbon Tax Over Time", fontsize=20)
plt.tight_layout()
plt.show()

# === Second Figure: Consumer Impact ===
fig, ax2 = plt.subplots(figsize=(14, 7))

ax2.plot(
    data_pct_reset["Time"], data_pct_reset["PctIncreasePriceOfGoods"],
    color="green", linewidth=2, label="Consumer Impact Increase (%)"
)
ax2.set_xlabel("Time Step", fontsize=20)
ax2.set_ylabel("Increase in Consumer Impact (%)", fontsize=20, color="green")
ax2.tick_params(axis="y", labelcolor="green", labelsize=15)
ax2.set_ylim(0, 140)
ax2.set_xlim(0, data_pct_reset["Time"].max())
ax2.grid(True, which='major', axis='both', linestyle='--', linewidth=0.7)

# Carbon Tax axis for this subplot as well
ax2b = ax2.twinx()
ax2b.plot(
    data_pct_reset["Time"], data_pct_reset["Tax"],
    color="blue", linestyle="--", linewidth=2, label="Carbon Tax"
)
ax2b.set_ylabel("Carbon Tax Level", fontsize=20, color="blue")
ax2b.tick_params(axis="y", labelcolor="blue", labelsize=15)
ax2b.set_ylim(0, 5.0)

# Legends
lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=15)

#plt.title("Percentage Increase in Consumer Impact and Carbon Tax Over Time", fontsize=20)
plt.tight_layout()
plt.show()

#file_path = "./EmissionsVsTaxData_Chi0.70_SEED42_MODEAVERAGE.txt"
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