def plot_abatement_analysis(results):

    sec, p = results[0]
    t0 = 1
    T = p.T  # End here

    fig = plt.figure(figsize=(10, 7))
    columns = 3
    rows = 2

    # fig, ([[ax0,ax1,ax4],[ax2, ax3,ax5]]) = plt.subplots(nrows=2, ncols=3, figsize=(8, 8))
    # axs = [ax0,ax1,ax2,ax3,ax4]

    axs = [0]

    i = 1
    for sc in results:
        if True:

            ax = fig.add_subplot(rows, columns, i)
            axs.append(ax)

            sec, p = sc
            axs[i].set_title(p.mode, fontsize=fs)

            ab_21, ab_1, ab_22, ab_tot = calc_abatement_analysis(sc)

            pal = ["#bc5090", "#ffa600", "#58508d", "#003f5c"]
            stacks = axs[i].stackplot(range(t0, p.T+1), ab_21, ab_1, ab_22,  labels=[
                                      "Compositional change", "Technology adoption", "Reduction of production"], alpha=0.8, colors=pal)
            axs[i].axhline(p.sec.E[t0]-p.E_max, color='black', ls='--',
                           label='Abatement Target')  # Abatement Target Line
            # axs[i].plot(range(t0,p.T),ab_tot,label="Total Abatement")
            axs[i].set_xlabel('Time (t)', fontsize=fs)
            axs[i].grid()
            axs[i].set_xlim([0, 300])
            axs[i].set_ylim([0, 0.7])

        i += 1
    print("here i am!")
    print(i)
    axs[1].set_ylabel("Abatement ($α$)", fontsize=fs)
    #axs[4].set_ylabel("Abatement ($α$)", fontsize=fs)
    axs[1].legend(loc='upper left')
    plt.tight_layout()
    #plt.savefig('Outputs/abatement.pdf')
    plt.show()
