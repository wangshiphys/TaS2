import matplotlib.pyplot as plt
import numpy as np

color_map = plt.get_cmap("tab10")
COLORS = color_map(range(color_map.N))
color0 = "black"
color1 = COLORS[3]

params0 = dict(Muc=0.350, Uc=0.700, tsc=0.100)
params1 = dict(Muc=0.100, Uc=0.700, tsc=0.100)
file_name0 = "None Interacting Information for " + ",".join(
    "{0}={1:.3f}".format(key, params0[key])
    for key in sorted(params0)
) + ".npz"
file_name1 = "None Interacting Information for " + ",".join(
    "{0}={1:.3f}".format(key, params1[key])
    for key in sorted(params1)
) + ".npz"

with np.load(file_name0) as data:
    omegas0 = data["omegas"]
    dos0 = data["dos"]
    GMKGIndices0 = data["GMKGIndices"]
    GMKGPathEs0 = data["GMKGPathEs"]

with np.load(file_name1) as data:
    omegas1 = data["omegas"]
    dos1 = data["dos"]
    GMKGIndices1 = data["GMKGIndices"]
    GMKGPathEs1 = data["GMKGPathEs"]

labels = (r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$")
fig, axes = plt.subplots(1, 2)
lines0 = axes[0].plot(
    GMKGPathEs0[:, 0::2], ls="solid", color=color0, alpha=0.8
)
lines1 = axes[0].plot(
    GMKGPathEs1[:, 0::2], ls="dashed", color=color1
)
axes[0].set_xlim(0, len(GMKGPathEs0) - 1)
axes[0].set_ylim(omegas0[0], omegas0[-1])
axes[0].set_xticks(GMKGIndices0)
axes[0].set_xticklabels(labels)
axes[0].set_ylabel(r"$E$", rotation="horizontal")
axes[0].grid(axis="x", ls="dashed")
axes[0].legend(
    [lines0[0], lines1[0]], [r"$\mu_c=0.35$", r"$\mu_c=0.10$"], loc="best",
    prop={"family": "monospace", "size": "xx-large"}
)
axes[0].text(
    1.0, 1.0, "(a)", ha="right", va="top", transform=axes[0].transAxes,
    fontfamily="monospace", fontsize="xx-large"
)

line0, = axes[1].plot(np.sum(dos0, axis=-1), omegas0, color=color0)
line1, = axes[1].plot(np.sum(dos1, axis=-1), omegas1, color=color1)
line2, = axes[1].plot(
    np.sum(dos0[:, 0:2], axis=-1), omegas0, ls="dashed", color=color0
)
line3, = axes[1].plot(
    np.sum(dos1[:, 0:2], axis=-1), omegas1, ls="dashed", color=color1
)
line4, = axes[1].plot(
    np.sum(dos0[:, 2:], axis=-1), omegas0, ls="dotted", color=color0
)
line5, = axes[1].plot(
    np.sum(dos1[:, 2:], axis=-1), omegas1, ls="dotted", color=color1
)
handles = [line0, line2, line4, line1, line3, line5]
labels = [
    r" DOS   for $\mu_c=0.35$",
    r"PDOS-C for $\mu_c=0.35$",
    r"PDOS-S for $\mu_c=0.35$",
    r" DOS   for $\mu_c=0.10$",
    r"PDOS-C for $\mu_c=0.10$",
    r"PDOS-S for $\mu_c=0.10$",
]
axes[1].legend(
    handles, labels, loc="best",
    prop={"family": "monospace", "size": "xx-large"}
)
axes[1].text(
    1.0, 1.0, "(b)", ha="right", va="top", transform=axes[1].transAxes,
    fontfamily="monospace", fontsize="xx-large"
)

plt.show()
plt.close("all")

