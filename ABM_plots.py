import numpy as np
import matplotlib.pyplot as plt

def plot_emissions_over_time(results):

    """
    Plots emissions over time along with the carbon tax level on the same graph
    using a dual-axis approach.

    Parameters:
    - results (list): The output of the run_model() function containing the sector (sec) and parameters (p).
    """
    sec, p = results  # Extract sector and parameters from results

    # Create time axis
    time_steps = np.arange(1, p.T+1)

    # Extract emissions and tax level data
    emissions = sec.E[1:p.T+1]
    tax_levels = p.reg.pe[1:p.T+1]

    # Create figure and primary y-axis for emissions
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Total Emissions", color='red')
    ax1.plot(time_steps, emissions, label="Total Emissions", color='red', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='red')

    # Create secondary y-axis for tax levels
    ax2 = ax1.twinx()
    ax2.set_ylabel("Carbon Tax Level", color='blue')
    ax2.plot(time_steps, tax_levels, label="Carbon Tax", color='blue', linestyle='dashed', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='blue')

    # Add vertical line for policy start
    ax1.axvline(x=p.t_start, color='black', linestyle='--', label='Policy Start')

    # Add title and legend
    fig.suptitle("Emissions and Carbon Tax Over Time")
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    plt.grid()
    plt.show()


