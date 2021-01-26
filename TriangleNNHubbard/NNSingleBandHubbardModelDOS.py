import matplotlib.pyplot as plt
import numpy as np

data_path = "data/eta=0.10/DOS/"
data_name_template = "DOS for cluster=StarOfDavid," \
                     "enum={0},t={1:.2f},U={2:.2f}.npz"

linewidth = 5
fontsize = 30
fig, axes = plt.subplots(1, 3, sharex="all", sharey="all")
for index, U in enumerate([10, 11, 14]):
    ticks = []
    baseline = 0.0
    ax = axes[index]
    for e_num in [13, 14, 15, 16,17, 18]:
        data_name = data_name_template.format(e_num, 1.0, U)
        try:
            with np.load(data_path + data_name) as data:
                omegas = data["omegas"]
                dos = data["dos"]
        except OSError:
            continue

        line, = ax.plot(omegas, dos + baseline, lw=linewidth)
        if index == 0:
            ax.text(
                20, baseline, "N={0}".format(e_num),
                va="bottom", ha="right", fontsize=0.8*fontsize,
            )
        if index == 0:
            sub_fig = "(a) U = {0}t".format(U)
        elif index == 1:
            sub_fig = "(b) U = {0}t".format(U)
        else:
            sub_fig = "(c) U = {0}t".format(U)
        ax.text(
            0.02, 0.98, sub_fig,
            ha="left", va="top", fontsize=fontsize, transform=ax.transAxes
        )
        ticks.append(baseline)
        baseline += 0.1
    ax.set_xlim(-8, 20)
    ax.set_ylim(0, 0.7)
    ax.set_yticks(ticks)
    ax.tick_params(axis="both", labelsize=0.8*fontsize)
    ax.set_xlabel(r"$\omega/t$", fontsize=fontsize)
    ax.grid(axis="y", ls="dashed", lw=linewidth/3)
axes[0].set_ylabel(r"$A(\omega)$", fontsize=fontsize)
plt.show()
# fig.savefig("NNSingleBandHubbardModelDOS.pdf", transparent=True)
plt.close("all")
