import numpy as np
import matplotlib.pyplot as plt

def plot_emissions_over_time(results):

    sec, p = results[0]  # Extract sector and parameters from results

    # Create time axis
    time_steps = np.arange(1, p.T+1)
    
    # Extract emissions data
    emissions = sec.E[1:p.T+1]
    
    # Plot emissions over time
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, emissions, label='Total Emissions', color='red', linewidth=2)
    plt.axvline(x=p.t_start, color='black', linestyle='--', label='Policy Start')
    plt.xlabel("Time Steps")
    plt.ylabel("Total Emissions")
    plt.title("Emissions Over Time Under Carbon Tax Policy")
    plt.legend()
    plt.grid(True)
    plt.show()
