"""
Plot the Density of states versus omegas
"""


import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import argrelextrema


data_path = "data/"
default_model_params = {
    "mu_c": 0.212,
    "t_sc": 0.162,
    "t_ss1": 0.150,
    "t_ss2": 0.091,
    "t_ss3": 0.072,
    "t_ss4": 0.050,
    "t_ss5": 0.042,
    "t_ss6": 0.042,
    "U": 0.0,
}


def Visualize(ax, anchor=True, extra_displace=0, **model_params):
    new_model_params = dict(default_model_params)
    new_model_params.update(model_params)
    file_name = ",".join(
        "{0}={1:.3f}".format(key, value)
        for key, value in new_model_params.items()
    ) + ".npz"

    with np.load(data_path + file_name) as data:
        omegas = data["omegas"]
        dos = data["dos"]

    if anchor:
        extremes = argrelextrema(dos, np.greater)
        delta = omegas[extremes][-2] + extra_displace
    else:
        delta = extra_displace
    line, = ax.plot(omegas - delta, dos)
    return line


if __name__ == "__main__":
    Mus = [0.35, 0.1]
    TSCs = [0.1, 0.12]
    Us = [0.7, 0.871]
    label_template = r"$\mu_c={0:.3f},t_{{sc}}={1:.3f},U={2:.3f}$"

    anchor = True
    fig, ax = plt.subplots()
    lines = []
    labels = []

    xmin = []
    xmax = []
    for mu_c, t_sc, U in zip(Mus, TSCs, Us):
        line = Visualize(ax, anchor, mu_c=mu_c, t_sc=t_sc, U=U)
        xdata = line.get_data()[0]
        xmin.append(xdata[0])
        xmax.append(xdata[-1])

        lines.append(line)
        labels.append(label_template.format(mu_c, t_sc, U))

    ax.set_xlim(max(xmin), min(xmax))
    ax.legend(lines, labels, fontsize="xx-large")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("DoS(a.u.)")
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()
    plt.close("all")
