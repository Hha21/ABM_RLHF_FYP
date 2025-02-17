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

# def plot_measures_over_time(results):
#     """
#     Plots multiple relevant measures over time.

#     Parameters:
#     - results (list): The output of the run_model() function containing the sector (sec) and parameters (p).
#     - measures_names (list of str): Names of the measures to plot.
#     - measures_values (list of lists): Corresponding values of the measures over time.
#     """

#     measures_names = ["Emissions", "Abatement \n Costs", "Emissions \n Costs", "Technology \n Adoption", "Compositional \n Change",
#                  "Product \n Sales", "Profit \n rate", "Market \n Concentration", "Sales \n Price", "Consumer \n Impact", "Emissions \n Price"]

#     measures_cum = []
#     sec, p = results[0]

#     for t in range(p.T+1):
#         evaluation_measures_cumulative(measures_cum, results, t)


#     measures_cum = np.array(measures_cum).T  # Transpose so each row corresponds to a measure

#     time_steps = np.arange(1, p.T+2)  # Create time axis

#     print(np.shape(time_steps))
#     print(np.shape(measures_cum))

#     plt.figure(figsize=(12, 6))

#     for i, measure in enumerate(measures_names):
#         plt.plot(time_steps, measures_cum[i], label=measure, linewidth=2)

#     plt.axvline(x=p.t_start, color='black', linestyle='dashed', label='Policy Start')
#     plt.xlabel("Time Steps")
#     plt.ylabel("Value")
#     plt.title("Cumulative Measures Over Time Under Carbon Tax Policy")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# def evaluation_measures_cumulative(measures_cum, results, t):

#     sec, p = results

#     # Effectiveness & Economic Impact
#     E = sum([sec.E[ti] for ti in range(t-p.t_period, t)])
#     CA = sum([sum([(j.B[ti]-j.B[1])*j.qg[ti] for j in sec])
#                 for ti in range(t-p.t_period, t)]) / E
#     CE = sum([sum([j.e[ti]*j.c_e[ti] for j in sec])
#                 for ti in range(t-p.t_period, t)])
#     HHI = sum([sum([j.s[ti]**2 for j in sec])
#                 for ti in range(t-p.t_period, t)])

#     # Efficiency / Abatement Decomposition
#     # "Compositional change","Technology adoption","Reduction of total production"
#     ac, at, ar, ab_tot = calc_abatement_analysis(sec)
#     AT = sum(at[t-p.t_period:t])
#     AC = sum(ac[t-p.t_period:t])

#     # Consumer Impact
#     S = sum([sum([j.qg_s[ti] for j in sec])
#             for ti in range(t-p.t_period, t)])
#     PL = sum([sum([j.qg_s[ti] * (j.pg[ti] - (j.c_e[ti] * j.A[ti] + j.B[ti])) for j in sec]) for ti in range(t-p.t_period, t)]
#                 ) / sum([sum([j.qg[ti] * (j.c_e[ti] * j.A[ti] + j.B[ti]) for j in sec]) for ti in range(t-p.t_period, t)])
#     CC0 = sum([sum([j.s[ti] * j.pg[ti] for j in sec])
#                 for ti in range(t-p.t_period, t)]) / 10
#     CC = sum([sum([j.s[ti] * j.pg[ti] for j in sec]) for ti in range(t-p.t_period, t)]
#                 ) / 10 - sum([p.reg.R[ti] for ti in range(t-p.t_period, t)]) / S
#     R = sum([p.reg.R[ti] for ti in range(t-p.t_period, t)])

#     # Others
#     PE = sum([p.reg.pe[ti] for ti in range(t-p.t_period, t)])

#     measures_cum.append([E, CA, CE, AT, AC,
#                     S, PL, HHI, CC0, CC, PE])

#     return measures_cum