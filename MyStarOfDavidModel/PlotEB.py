from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

eta = 0.10
data_path = "data/EB/eta={0:.2f}/".format(eta)
data_name_template = "EB for cluster=StarOfDavid,enum={0},{1}.npz"

fig_path = "fig/EB/eta={0:.2f}/".format(eta)
Path(fig_path).mkdir(exist_ok=True, parents=True)
fig_name_template = "EB for cluster=StarOfDavid,enum={0},{1}.jpg"

title_template = r"$N={N},\mu_0={mu0: .2f},\mu_1={mu1: .2f},\mu_2={mu2: .2f}," \
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

e_nums = [13, 14]
Us = np.arange(0.0, 2.1, 0.1)
for U in Us:
    new_model_params = dict(default_model_parameters)
    new_model_params.update(U=U)
    tmp = ",".join(
        "{0}={1:.2f}".format(key, value)
        for key, value in new_model_params.items()
    )
    for e_num in e_nums:
        data_name = data_name_template.format(e_num, tmp)
        fig_name = fig_name_template.format(e_num, tmp)
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
        ax.set_title(
            title_template.format(N=e_num, **new_model_params),
            fontsize="x-large"
        )
        ax.set_xticks(xticks)
        ax.set_xticklabels([r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$"])
        ax.grid(axis="x", ls="dashed")
        fig.colorbar(cs, ax=ax)
        plt.get_current_fig_manager().window.showMaximized()
        plt.tight_layout()
        # plt.show()
        fig.savefig(fig_path + fig_name, dpi=300)
        plt.close("all")
