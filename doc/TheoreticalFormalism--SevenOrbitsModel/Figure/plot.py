import matplotlib.pyplot as plt
import numpy as np


color0 = "black"
color1 = "red"
LINEWIDTH = 5
FONTSIZE = "x-large"
DPI = 100

prefix0 = "enum=13/" + "None Interacting Information for "
prefix1 = "enum=13/" + "DOS for "
prefix2 = "enum=11/" + "None Interacting Information for "
prefix3 = "enum=11/" + "DOS for "
tmp = "Muc={Muc:.3f},Uc={Uc:.3f},tsc={tsc:.3f}.npz"

params0 = dict(Muc=0.350, Uc=0.700, tsc=0.100)
params1 = dict(Muc=0.100, Uc=0.700, tsc=0.100)
params2 = dict(Muc=0.200, Uc=0.500, tsc=0.010)
params3 = dict(Muc=0.200, Uc=0.500, tsc=0.050)

fig = plt.figure(constrained_layout=False)
gs0 = fig.add_gridspec(2, 1, left=0.05, right=0.20, hspace=0.1)
gs1 = fig.add_gridspec(2, 1, left=0.20, right=0.50, hspace=0.1)
gs2 = fig.add_gridspec(2, 1, left=0.52, right=0.99, hspace=0.1)
ax0 = fig.add_subplot(gs0[0, 0])
ax1 = fig.add_subplot(gs1[0, 0])
ax2 = fig.add_subplot(gs2[0, 0])
ax3 = fig.add_subplot(gs0[1, 0])
ax4 = fig.add_subplot(gs1[1, 0])
ax5 = fig.add_subplot(gs2[1, 0])


with np.load(prefix0 + tmp.format(**params0)) as data:
    omegas0 = data["omegas"]
    dos0 = data["dos"]
    GMKGIndices0 = data["GMKGIndices"]
    GMKGPathEs0 = data["GMKGPathEs"]

with np.load(prefix0 + tmp.format(**params1)) as data:
    omegas1 = data["omegas"]
    dos1 = data["dos"]
    GMKGIndices1 = data["GMKGIndices"]
    GMKGPathEs1 = data["GMKGPathEs"]

EBLines0 = ax0.plot(
    GMKGPathEs0[:, ::2], color=color0, lw=LINEWIDTH, ls="solid",
)
EBLines1 = ax0.plot(
    GMKGPathEs1[:, ::2], color=color1, lw=LINEWIDTH, ls="dashed",
)
ax0.set_xlim(0, len(GMKGPathEs0) - 1)
ax0.set_ylim(max([omegas0[0], omegas1[0]]), min([omegas0[-1], omegas1[-1]]))
ax0.set_xticks(GMKGIndices0)
ax0.tick_params(labelbottom=False, labelsize=FONTSIZE)
# ax0.set_xticklabels((r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$"))
# ax0.tick_params(labelsize=FONTSIZE)
ax0.set_ylabel(r"E(eV)", fontsize=FONTSIZE, labelpad=0)
ax0.grid(axis="x", ls="dashed")
handles = [EBLines0[0], EBLines1[0]]
labels = [
    r"$\mu_c={Muc:.2f}$".format(**params0),
    r"$\mu_c={Muc:.2f}$".format(**params1),
]
ax0.legend(
    handles, labels, fontsize=FONTSIZE,
    loc="lower right", bbox_to_anchor=(1.0, 0.48),
)
ax0.text(
    1.0, 1.0, "(b)",
    ha="right", va="top", transform=ax0.transAxes, fontsize=FONTSIZE
)

DOSLine0, = ax1.plot(
    np.sum(dos0, axis=-1), omegas0,
    color=color0, lw=LINEWIDTH, ls="solid"
)
DOSLine1, = ax1.plot(
    np.sum(dos0[:, 0:2], axis=-1), omegas0,
    color=color0, lw=LINEWIDTH, ls="dotted"
)
DOSLine2, = ax1.plot(
    np.sum(dos1, axis=-1), omegas1,
    color=color1, lw=LINEWIDTH, ls="solid"
)
DOSLine3, = ax1.plot(
    np.sum(dos1[:, 0:2], axis=-1), omegas1,
    color=color1, lw=LINEWIDTH, ls="dotted"
)
ax1.set_ylim(max([omegas0[0], omegas1[0]]), min([omegas0[-1], omegas1[-1]]))
ax1.tick_params(labelbottom=False, labelleft=False)
# ax1.set_xlabel("DOS(arb. units)", fontsize=FONTSIZE)
handles = [DOSLine0, DOSLine1, DOSLine2, DOSLine3]
labels = [
    r"$\mu_c={Muc:.2f}$, DOS".format(**params0),
    r"$\mu_c={Muc:.2f}$, PDOS-C".format(**params0),
    r"$\mu_c={Muc:.2f}$, DOS".format(**params1),
    r"$\mu_c={Muc:.2f}$, PDOS-C".format(**params1),
]
ax1.legend(
    handles, labels, fontsize=FONTSIZE,
    loc="lower right", bbox_to_anchor=(1.0, 0.47),
)
ax1.text(
    1.0, 1.0, "(c)",
    ha="right", va="top", transform=ax1.transAxes, fontsize=FONTSIZE
)

with np.load(prefix1 + tmp.format(**params0)) as data:
    omegas0 = data["omegas"]
    dos0 = data["dos"]
with np.load(prefix1 + tmp.format(**params1)) as data:
    omegas1 = data["omegas"]
    dos1 = data["dos"]

line0, = ax2.plot(omegas0, dos0, color=color0, lw=LINEWIDTH)
line1, = ax2.plot(omegas1 + 0.09, dos1, color=color1, lw=LINEWIDTH)
handles = [line0, line1]
labels = [
    r"$\mu_c={Muc:.2f}$".format(**params0),
    r"$\mu_c={Muc:.2f}$".format(**params1),
]
ax2.legend(
    handles, labels, fontsize=FONTSIZE,
    loc="upper right", bbox_to_anchor=(0.96, 1.00),
)
ax2.text(
    1.0, 1.0, "(d)",
    ha="right", va="top", transform=ax2.transAxes, fontsize=FONTSIZE
)
ax2.set_xlim(0.0, 1.3)
ax2.tick_params(axis="x", labelsize=FONTSIZE)
ax2.tick_params(axis="y", labelleft=False)
# ax2.set_xlabel(r"$\omega(eV)$", fontsize=FONTSIZE)
ax2.set_ylabel("DOS (arb. units)", fontsize=FONTSIZE)


with np.load(prefix2 + tmp.format(**params2)) as data:
    omegas0 = data["omegas"]
    dos0 = data["dos"]
    GMKGIndices0 = data["GMKGIndices"]
    GMKGPathEs0 = data["GMKGPathEs"]

with np.load(prefix2 + tmp.format(**params3)) as data:
    omegas1 = data["omegas"]
    dos1 = data["dos"]
    GMKGIndices1 = data["GMKGIndices"]
    GMKGPathEs1 = data["GMKGPathEs"]

EBLines0 = ax3.plot(
    GMKGPathEs0[:, ::2], color=color0, lw=LINEWIDTH, ls="solid",
)
EBLines1 = ax3.plot(
    GMKGPathEs1[:, ::2], color=color1, lw=LINEWIDTH, ls="dashed",
)
ax3.set_xlim(0, len(GMKGPathEs0) - 1)
ax3.set_ylim(max([omegas0[0], omegas1[0]]), min([omegas0[-1], omegas1[-1]]))
ax3.set_xticks(GMKGIndices0)
ax3.set_xticklabels((r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$"))
ax3.tick_params(labelsize=FONTSIZE)
ax3.set_ylabel(r"E(eV)", fontsize=FONTSIZE, labelpad=0)
ax3.grid(axis="x", ls="dashed")
handles = [EBLines0[0], EBLines1[0]]
labels = [
    r"$t_{{sc}}={tsc:.2f}$".format(**params0),
    r"$t_{{sc}}={tsc:.2f}$".format(**params1),
]
ax3.legend(
    handles, labels, fontsize=FONTSIZE,
    loc="upper center", bbox_to_anchor=(0.5, 1.0),
)
ax3.text(
    1.0, 1.0, "(e)",
    ha="right", va="top", transform=ax3.transAxes, fontsize=FONTSIZE
)

DOSLine0, = ax4.plot(
    np.sum(dos0, axis=-1), omegas0,
    color=color0, lw=LINEWIDTH, ls="solid"
)
DOSLine1, = ax4.plot(
    np.sum(dos0[:, 0:2], axis=-1), omegas0,
    color=color0, lw=LINEWIDTH, ls="dotted"
)
DOSLine2, = ax4.plot(
    np.sum(dos1, axis=-1), omegas1,
    color=color1, lw=LINEWIDTH, ls="solid"
)
DOSLine3, = ax4.plot(
    np.sum(dos1[:, 0:2], axis=-1), omegas1,
    color=color1, lw=LINEWIDTH, ls="dotted"
)
ax4.set_ylim(max([omegas0[0], omegas1[0]]), min([omegas0[-1], omegas1[-1]]))
ax4.tick_params(labelbottom=False, labelleft=False)
ax4.set_xlabel("DOS(arb. units)", fontsize=FONTSIZE)
handles = [DOSLine0, DOSLine1, DOSLine2, DOSLine3]
labels = [
    r"$t_{{sc}}={tsc:.2f}$, DOS".format(**params0),
    r"$t_{{sc}}={tsc:.2f}$, PDOS-C".format(**params0),
    r"$t_{{sc}}={tsc:.2f}$, DOS".format(**params1),
    r"$t_{{sc}}={tsc:.2f}$, PDOS-C".format(**params1),
]
ax4.legend(
    handles, labels, fontsize=FONTSIZE,
    loc="upper right", bbox_to_anchor=(0.96, 1.0),
)
ax4.text(
    1.0, 1.0, "(f)",
    ha="right", va="top", transform=ax4.transAxes, fontsize=FONTSIZE
)

with np.load(prefix3 + tmp.format(**params2)) as data:
    omegas0 = data["omegas"]
    dos0 = data["dos"]
with np.load(prefix3 + tmp.format(**params3)) as data:
    omegas1 = data["omegas"]
    dos1 = data["dos"]

line0, = ax5.plot(omegas0, dos0, color=color0, lw=LINEWIDTH)
line1, = ax5.plot(omegas1 + 0.06, dos1, color=color1, lw=LINEWIDTH)
handles = [line0, line1]
labels = [
    r"$t_{{sc}}={tsc:.2f}$".format(**params0),
    r"$t_{{sc}}={tsc:.2f}$".format(**params1),
]
ax5.legend(
    handles, labels, fontsize=FONTSIZE,
    loc="upper right", bbox_to_anchor=(0.96, 1.0),
)
ax5.text(
    1.0, 1.0, "(g)",
    ha="right", va="top", transform=ax5.transAxes, fontsize=FONTSIZE
)
ax5.set_xlim(0.04, 1.0)
ax5.tick_params(axis="x", labelsize=FONTSIZE)
ax5.tick_params(axis="y", labelleft=False)
ax5.set_xlabel(r"$\omega(eV)$", fontsize=FONTSIZE)
ax5.set_ylabel("DOS (arb. units)", fontsize=FONTSIZE)

fig.subplots_adjust(bottom=0.07, top=0.99)
plt.show()