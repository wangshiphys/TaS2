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
    # return muh

color0 = "black"
color1 = "red"
LINEWIDTH = 5
FONTSIZE = 22
dpi = 100

prefix0 = "None Interacting Information for "
prefix1 = "DOS for "
tmp = "Muc={Muc:.3f},Uc={Uc:.3f},tsc={tsc:.3f}.npz"

params0 = dict(Muc=0.200, Uc=0.500, tsc=0.010)
params1 = dict(Muc=0.200, Uc=0.500, tsc=0.050)

fig = plt.figure(constrained_layout=False)
gs0 = fig.add_gridspec(1, 1, left=0.06, right=0.26)
gs1 = fig.add_gridspec(1, 1, left=0.26, right=0.60)
gs2 = fig.add_gridspec(1, 1, left=0.63, right=0.985)
ax0 = fig.add_subplot(gs0[0])
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs2[0])

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
ax0.set_xticklabels((r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$"))
ax0.tick_params(labelsize=FONTSIZE)
ax0.set_ylabel(r"E(eV)", fontsize=FONTSIZE, labelpad=0)
ax0.grid(axis="x", ls="dashed")
handles = [EBLines0[0], EBLines1[0]]
labels = [
    r"$t_{{sc}}={tsc:.2f}$".format(**params0),
    r"$t_{{sc}}={tsc:.2f}$".format(**params1),
]
ax0.legend(
    handles, labels, fontsize=FONTSIZE,
    loc="upper center", bbox_to_anchor=(0.5, 1.0),
)
ax0.text(
    1.0, 1.0, "(e)",
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
ax1.set_xlabel("DOS(arb. units)", fontsize=FONTSIZE)
handles = [DOSLine0, DOSLine1, DOSLine2, DOSLine3]
labels = [
    r"$t_{{sc}}={tsc:.2f}$, DOS".format(**params0),
    r"$t_{{sc}}={tsc:.2f}$, PDOS-C".format(**params0),
    r"$t_{{sc}}={tsc:.2f}$, DOS".format(**params1),
    r"$t_{{sc}}={tsc:.2f}$, PDOS-C".format(**params1),
]
ax1.legend(
    handles, labels, fontsize=FONTSIZE,
    loc="upper right", bbox_to_anchor=(0.96, 1.0),
)
ax1.text(
    1.0, 1.0, "(f)",
    ha="right", va="top", transform=ax1.transAxes, fontsize=FONTSIZE
)

with np.load(prefix1 + tmp.format(**params0)) as data:
    omegas0 = data["omegas"]
    dos0 = data["dos"]
    mu0 = Mu(dos0, omegas0, occupied_num=11)
    print("mu0 = {}".format(mu0))
with np.load(prefix1 + tmp.format(**params1)) as data:
    omegas1 = data["omegas"]
    dos1 = data["dos"]
    mu1 = Mu(dos1, omegas1, occupied_num=11)
    print("mu1 = {}".format(mu1))

line0, = ax2.plot(omegas0 - mu0, dos0, color=color0, lw=LINEWIDTH)
line1, = ax2.plot(omegas1 - mu1, dos1, color=color1, lw=LINEWIDTH)
ax2.axvline(0.0, ls="dashed", color="gray")
handles = [line0, line1]
labels = [
    r"$t_{{sc}}={tsc:.2f}$".format(**params0),
    r"$t_{{sc}}={tsc:.2f}$".format(**params1),
]
ax2.legend(
    handles, labels, fontsize=FONTSIZE,
    loc="upper right", bbox_to_anchor=(0.96, 1.0),
)
ax2.text(
    1.0, 1.0, "(g)",
    ha="right", va="top", transform=ax2.transAxes, fontsize=FONTSIZE
)
# ax2.set_xlim(0.04, 1.0)
ax2.set_xlim(-0.3, 0.75)

ax2.tick_params(axis="x", labelsize=FONTSIZE)
ax2.tick_params(axis="y", labelleft=False)
ax2.set_xlabel(r"$\omega(eV)$", fontsize=FONTSIZE)
ax2.set_ylabel("DOS (arb. units)", fontsize=FONTSIZE)
fig.subplots_adjust(bottom=0.08, top=0.99)
plt.show()
# fig.savefig("ENUM=11.jpg", dpi=dpi)
# fig.savefig("ENUM=11.png", dpi=dpi)
# fig.savefig("ENUM=11.pdf", dpi=dpi)
plt.close("all")
