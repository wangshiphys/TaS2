from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

data_path = "data/eta=0.10/DOS/"
data_name_template = "DOS for cluster=StarOfDavid," \
                     "enum={0},t={1:.2f},U={2:.2f}.npz"

fig_path = "fig/eta=0.10/DOS/"
Path(fig_path).mkdir(exist_ok=True, parents=True)
fig_name_template = "DOS for cluster=StarOfDavid," \
                     "enum={0},t={1:.2f},U={2:.2f}.jpg"

title_template = "DOS for cluster=StarOfDavid," \
                 "enum={0},t={1:.2f},U={2:.2f}"

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
                dos = data["dos"]
        except OSError:
            continue

        fig, ax = plt.subplots()
        ax.plot(omegas, dos)
        ax.set_title(title_template.format(e_num, t, U), fontsize="xx-large")
        plt.get_current_fig_manager().window.showMaximized()
        plt.tight_layout()
        plt.show()
        fig.savefig(fig_path + fig_name, dpi=300)
        plt.close("all")
