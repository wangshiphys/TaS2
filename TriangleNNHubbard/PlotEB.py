from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

data_path = "data/eta=0.10/EB/"
data_name_template = "EB for cluster=StarOfDavid," \
                     "enum={0},t={1:.2f},U={2:.2f}.npz"

fig_path = "fig/eta=0.10/EB/"
Path(fig_path).mkdir(exist_ok=True, parents=True)
fig_name_template = "EB for cluster=StarOfDavid," \
                     "enum={0},t={1:.2f},U={2:.2f}.jpg"

title_template = "EB for cluster=StarOfDavid," \
                 "enum={0},t={1:.2f},U={2:.2f}"
xlables = [r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$"]

t = 1.0
state_num = 26
e_nums = tuple(range(state_num))
Us = np.arange(0.0, 20.0, 0.1)
for e_num in e_nums:
    for U in Us:
        fig_name = fig_name_template.format(e_num, t, U)
        data_name = data_name_template.format(e_num, t, U)
        try:
            with np.load(data_path + data_name) as data:
                omegas = data["omegas"]
                spectrums = data["spectrums"]
        except OSError:
            continue

        k_num = spectrums.shape[1]
        xticks = [0, k_num//3, 2*k_num//3, k_num - 1]

        fig, ax = plt.subplots()
        cs = ax.contourf(
            range(k_num), omegas, spectrums, levels=200, cmap="hot"
        )
        ax.set_title(title_template.format(e_num, t, U), fontsize="xx-large")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlables)
        ax.grid(axis="x", ls="dashed")

        fig.colorbar(cs, ax=ax)
        plt.get_current_fig_manager().window.showMaximized()
        plt.tight_layout()
        plt.show()
        fig.savefig(fig_path + fig_name, transparent=True)
        plt.close("all")
