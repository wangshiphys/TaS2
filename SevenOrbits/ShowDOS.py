"""
Plot the Density-of-states versus omegas
"""


import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import brentq


def _MuCore(mu, dos, omegas, occupied_num=13, total_num=14, reverse=False):
    delta_omega = omegas[1] - omegas[0]
    if reverse:
        num = total_num - occupied_num
        indices = omegas > mu
    else:
        num = occupied_num
        indices = omegas < mu
    return np.sum(dos[indices]) * delta_omega - num


def Mu(dos, omegas, occupied_num=13, total_num=14):
    args0 = (dos, omegas, occupied_num, total_num, False)
    args1 = (dos, omegas, occupied_num, total_num, True)
    mup = brentq(_MuCore, a=omegas[0], b=omegas[-1], args=args0)
    muh = brentq(_MuCore, a=omegas[0], b=omegas[-1], args=args1)
    return (mup + muh) / 2


def Visualize(ax, e_num=13, eta=0.01, **model_params):
    data_path = "data/e_num={0},eta={1:.3f}/".format(e_num, eta)
    file_name = "DOS for " + ",".join(
        "{0}={1:.3f}".format(key, model_params[key])
        for key in sorted(model_params)
    ) + ".npz"

    with np.load(data_path + file_name) as data:
        omegas = data["omegas"]
        dos = data["dos"]
        mu = Mu(dos, omegas)
        data = np.array([omegas, dos]).T
        np.savetxt(file_name.replace("npz", "dat"), data)

    line, = ax.plot(omegas - mu, dos)
    ax.axvline(0.0, ls="dashed", color="gray")
    return line


if __name__ == "__main__":
    e_num = 13
    params0 = dict(Muc=0.350, Uc=0.700, tsc=0.100)
    params1 = dict(Muc=0.100, Uc=0.700, tsc=0.100)
    params2 = dict(Muc=0.000, Uc=0.700, tsc=0.100)
    params3 = dict(Muc=-0.100, Uc=0.700, tsc=0.100)
    params4 = dict(Muc=-0.200, Uc=0.700, tsc=0.100)
    all_cases = [
        (0.00, e_num, params0),
        (0.00, e_num, params1),
        (0.00, e_num, params2),
        (0.00, e_num, params3),
        (0.00, e_num, params4),
    ]

    lines = []
    labels = []
    fig, ax = plt.subplots()
    for displace, e_num, params in all_cases:
        lines.append(Visualize(ax, e_num, **params))
        label = ",".join(
            "{0}={1:.3f}".format(key, params[key])
            for key in sorted(params)
        )
        labels.append(label)

    ax.grid(axis="y")
    ax.legend(lines, labels, fontsize="xx-large")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("DOS(a.u.)")
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()
    plt.close("all")
