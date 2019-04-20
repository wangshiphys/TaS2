"""
Plot the Density-of-states versus omegas
"""


import matplotlib.pyplot as plt
import numpy as np


def Visualize(ax, displace=0.0, eta=0.01, **model_params):
    data_path = "data/eta={0:.3f}/".format(eta)
    file_name = "DOS at " + ",".join(
        "{0}={1:.3f}".format(key, model_params[key])
        for key in sorted(model_params)
    ) + ".npz"

    with np.load(data_path + file_name) as data:
        omegas = data["omegas"]
        dos = data["dos"]

    line, = ax.plot(omegas + displace, dos)
    return line


if __name__ == "__main__":
    label_template = r"$\mu_c={Muc:.3f},U_c={Uc:.3f}," \
                     r"U_{{sc}}={Usc:.3f},t_{{sc}}={tsc:.3f}$"

    model_params0 = dict(Muc=0.350, Uc=0.700, Usc=0.01, tsc=0.100,)
    model_params1 = dict(Muc=0.100, Uc=0.871, Usc=0.01, tsc=0.120,)
    all_cases = [
        (0.0, model_params0),
        (0.0, model_params1),
    ]

    lines = []
    labels = []
    xmin = []
    xmax = []
    fig, ax = plt.subplots()
    for displace, model_params in all_cases:
        line = Visualize(ax, displace, **model_params)
        lines.append(line)
        labels.append(label_template.format(**model_params))

        xdata = line.get_data()[0]
        xmin.append(xdata[0])
        xmax.append(xdata[-1])

    ax.set_xlim(max(xmin), min(xmax))
    ax.legend(lines, labels, fontsize="xx-large")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("DOS(a.u.)")
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()
    plt.close("all")
