from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


eta = 0.14
data_path = "data/DOS/eta={0:.2f}/".format(eta)
data_name_template = "DOS for cluster=StarOfDavid,enum={0},{1}.npz"

fig_path = "fig/ComposedDOS/eta={0:.2f}/".format(eta)
Path(fig_path).mkdir(exist_ok=True, parents=True)
fig_name_template = "Composed DOS for cluster=StarOfDavid,{0}.jpg"

title_template = r"$\mu_0={mu0: .2f},\mu_1={mu1: .2f},\mu_2={mu2: .2f}," \
                 r"t_0={t0: .2f},t_1={t1: .2f},t_2={t2: .2f}," \
                 r"t_3={t3: .2f},t_4={t4: .2f},t_5={t5: .2f},U={U: .2f}$"


default_model_parameters = {
    "t0" : -0.2,
    "t1" : -0.2,
    "t2" : -0.8,
    "t3" : -1.0,
    "t4" : -1.0,
    "t5" : -1.0,
    "mu0":  0.4,
    "mu1": -0.2,
    "mu2":  0.1,
    "U"  :  0.0,
}

U = 1.6
e_nums = [13, 14]
new_model_params = dict(default_model_parameters)
new_model_params.update(U=U)
tmp = ",".join(
    "{0}={1:.2f}".format(key, value) for key, value in new_model_params.items()
)

labels = []
handles = []
fig, ax = plt.subplots()
fig_name = fig_name_template.format(tmp)
for e_num in e_nums:
    data_name = data_name_template.format(e_num, tmp)
    try:
        with np.load(data_path + data_name) as data:
            omegas = data["omegas"]
            dos = data["dos"]
    except OSError:
        continue

    line, = ax.plot(omegas, dos)
    handles.append(line)
    labels.append("N={0}".format(e_num))

ax.legend(handles, labels, loc="upper left")
ax.set_title(title_template.format(**new_model_params), fontsize="x-large")
plt.get_current_fig_manager().window.showMaximized()
plt.show()
fig.savefig(fig_path + fig_name, dpi=300)
plt.close("all")
