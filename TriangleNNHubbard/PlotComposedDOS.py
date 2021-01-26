from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

data_path = "data/eta=0.10/DOS/"
data_name_template = "DOS for cluster=StarOfDavid," \
                     "enum={0},t={1:.2f},U={2:.2f}.npz"

fig_path = "fig/eta=0.10/ComposedDOS/"
Path(fig_path).mkdir(exist_ok=True, parents=True)
fig_name_template = "Composed DOS for cluster=StarOfDavid," \
                    "t={0:.2f},U={1:.2f}.jpg"
title_template = "Composed DOS for cluster=StarOfDavid," \
                    "t={0:.2f},U={1:.2f}"

t = 1.0
DY = 0.1
state_num = 26
Us = range(8, 15)
e_nums = range(13, state_num)
for U in Us:
    dy = -DY
    ticks = []
    labels = []
    handles = []
    fig, ax = plt.subplots()
    fig_name = fig_name_template.format(t, U)
    for index, e_num in enumerate(e_nums):
        data_name = data_name_template.format(e_num, t, U)
        try:
            with np.load(data_path + data_name) as data:
                omegas = data["omegas"]
                dos = data["dos"]
        except OSError:
            continue

        dy += DY
        line, = ax.plot(omegas, dos + dy)
        ticks.append(dy)
        handles.append(line)
        labels.append("N={0}".format(e_num))

    ax.legend(handles[::-1], labels[::-1], loc="upper left")
    ax.set_xlim(-8, 18)
    ax.set_yticks(ticks)
    ax.grid(axis="y")
    ax.set_title(title_template.format(t, U), fontsize="xx-large")
    plt.get_current_fig_manager().window.showMaximized()
    plt.tight_layout()
    plt.show()
    fig.savefig(fig_path + fig_name, dpi=300)
    plt.close("all")
