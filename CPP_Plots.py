import matplotlib.pyplot as plt
import pandas as pd

#file_path = "./EmissionsVsTaxData_Chi0.90_SEED42_MODEAVERAGE.txt"
file_path = "./EmissionsVsTaxData_Chi0.50_SEED42_MODEPESSIMISTIC.txt"

data = pd.read_csv(
    file_path,
    delim_whitespace=True,
    skiprows=1,
    names=["Emissions", "Tax", "Price of Goods"]
)
data["Time"] = range(1, len(data) + 1)

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

# Reset time to start from 1
initial_row = pd.DataFrame({
    "Time": [0],
    "PctDecreaseEmissions": [0],
    "PctIncreasePriceOfGoods": [0],
    "Tax": [0]
})
data_pct_reset = pd.concat([initial_row, data_pct[["Time", "PctDecreaseEmissions", "PctIncreasePriceOfGoods", "Tax"]]], ignore_index=True)
data_pct_reset["Time"] = range(1, len(data_pct_reset) + 1)

fig1, ax1 = plt.subplots(figsize=(14, 6))
color_em = "red"
color_tax = "blue"
ax1.plot(
    data_pct_reset["Time"], data_pct_reset["PctDecreaseEmissions"],
    color=color_em,
    label="Emissions decrease (%)"
)
ax1.set_title("Percentage Decrease in Emissions Over Time")
ax1.set_xlabel("Time Step")
ax1.set_ylabel("Decrease in Emissions (%)", color=color_em)
ax1.tick_params(axis="y", labelcolor=color_em)

ax1b = ax1.twinx()
ax1b.plot(
    data_pct_reset["Time"], data_pct_reset["Tax"],
    color=color_tax, linestyle="--",
    label="Carbon Tax"
)
ax1b.set_ylabel("Carbon Tax Level", color=color_tax)
ax1b.tick_params(axis="y", labelcolor=color_tax)
fig1.tight_layout()

fig2, ax2 = plt.subplots(figsize=(14, 6))
color_pg = "green"
ax2.plot(
    data_pct_reset["Time"], data_pct_reset["PctIncreasePriceOfGoods"],
    color=color_pg,
    label="Consumer impact increase (%)"
)
ax2.set_title("Percentage Increase in Consumer Impact Over Time")
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Increase in Price of Goods (%)", color=color_pg)
ax2.tick_params(axis="y", labelcolor=color_pg)

ax2b = ax2.twinx()
ax2b.plot(
    data_pct_reset["Time"], data_pct_reset["Tax"],
    color=color_tax, linestyle="--",
    label="Carbon Tax"
)
ax2b.set_ylabel("Carbon Tax Level", color=color_tax)
ax2b.tick_params(axis="y", labelcolor=color_tax)
fig2.tight_layout()

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