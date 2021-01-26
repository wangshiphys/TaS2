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
    # return mup
    # return muh



color0 = "black"
color1 = "red"
LINEWIDTH = 5
FONTSIZE = 22
dpi = 100

prefix0 = "None Interacting Information for "
prefix1 = "DOS for "
tmp = "Muc={Muc:.3f},Uc={Uc:.3f},tsc={tsc:.3f}.npz"

params0 = dict(Muc=0.350, Uc=0.700, tsc=0.100)
params1 = dict(Muc=0.100, Uc=0.700, tsc=0.100)

fig = plt.figure(constrained_layout=False)
gs0 = fig.add_gridspec(1, 1, left=0.06, right=0.26)
gs1 = fig.add_gridspec(1, 1, left=0.26, right=0.60)
gs2 = fig.add_gridspec(1, 1, left=0.63, right=0.99)
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
ax0.set_ylabel("E/eV", fontsize=FONTSIZE, labelpad=0)
ax0.grid(axis="x", ls="dashed")
handles = [EBLines0[0], EBLines1[0]]
labels = [
    r"$\mu_c={Muc:.2f}$".format(**params0),
    r"$\mu_c={Muc:.2f}$".format(**params1),
]
ax0.legend(
    handles, labels, fontsize=FONTSIZE,
    loc="lower right", bbox_to_anchor=(1.0, 0.52),
)
ax0.text(
    0.90, 0.98, "(a)",
    ha="right", va="top", transform=ax0.transAxes, fontsize=FONTSIZE+8
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
    0.90, 0.98, "(b)",
    ha="right", va="top", transform=ax1.transAxes, fontsize=FONTSIZE+8
)

with np.load(prefix1 + tmp.format(**params0)) as data:
    omegas0 = data["omegas"]
    dos0 = data["dos"]
    mu0 = Mu(dos0, omegas0, occupied_num=13)
    print("mu0 = {}".format(mu0))
with np.load(prefix1 + tmp.format(**params1)) as data:
    omegas1 = data["omegas"]
    dos1 = data["dos"]
    mu1 = Mu(dos1, omegas1, occupied_num=13)
    print("mu1 = {}".format(mu1))

line0, = ax2.plot(omegas0, dos0, color=color0, lw=LINEWIDTH)
line1, = ax2.plot(omegas1 + 0.09, dos1, color=color1, lw=LINEWIDTH)
# line0, = ax2.plot(omegas0 - mu0, dos0, color=color0, lw=LINEWIDTH, zorder=1)
# line1, = ax2.plot(omegas1 - mu1, dos1, color=color1, lw=LINEWIDTH, zorder=2)
# ax2.axvline(0.0, ls="dashed", color="gray", zorder=0)

ax2.annotate(
    "CB", xy=(0.28, 13.6), xytext=(0.3, 18.6),
    arrowprops={"arrowstyle": "->", "lw":3},
    fontsize=FONTSIZE, va="bottom", ha="center"
)
ax2.annotate(
    "", xy=(0.34, 14.6), xytext=(0.3, 18.6),
    arrowprops={"arrowstyle": "->", "lw":3, "color":"red"},
    fontsize=FONTSIZE, va="bottom", ha="center"
)
ax2.text(
    0.68, 12.7, "LHB", fontsize=FONTSIZE, va="bottom", ha="center"
)
ax2.annotate(
    "UHB", xy=(1.00, 18.2), xytext=(1.02, 24.5),
    arrowprops={"arrowstyle": "->", "lw":3, "color":"red"},
    fontsize=FONTSIZE, va="bottom", ha="center"
)
ax2.annotate(
    "", xy=(1.13, 22.5), xytext=(1.02, 24.5),
    arrowprops={"arrowstyle": "->", "lw":3},
    fontsize=FONTSIZE, va="bottom", ha="center"
)

handles = [line0, line1]
labels = [
    r"$\mu_c={Muc:.2f}$".format(**params0),
    r"$\mu_c={Muc:.2f}$".format(**params1),
]
ax2.legend(
    handles, labels, fontsize=FONTSIZE,
    loc="upper center", bbox_to_anchor=(0.50, 0.95),
)
ax2.text(
    0.90, 0.98, "(c)",
    ha="right", va="top", transform=ax2.transAxes, fontsize=FONTSIZE+8
)
ax2.set_xlim(0.0, 1.3)
# ax2.set_xlim(-1.03, 0.3)

ax2.tick_params(axis="x", labelsize=FONTSIZE)
ax2.tick_params(axis="y", labelleft=False)
ax2.set_xlabel(r"$\omega/eV$", fontsize=FONTSIZE)
ax2.set_ylabel("DOS (arb. units)", fontsize=FONTSIZE)
fig.subplots_adjust(bottom=0.08, top=0.99)
plt.show()
# fig.savefig("ENUM=13.jpg", dpi=dpi)
# fig.savefig("ENUM=13.png", dpi=dpi)
# fig.savefig("ENUM=13.pdf", dpi=dpi)
fig.savefig("MultiBandHubbardModel.pdf", transparent=True)
plt.close("all")
