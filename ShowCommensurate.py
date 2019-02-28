"""
A demonstration of commensurate/incommensurate in 1D for two periodical
system
"""


import matplotlib.pyplot as plt
import numpy as np

from TaS2DataBase import LINE_WIDTH, MARKER_SIZE, FONT_SIZE, FIG_DPI
from TaS2DataBase import SPINE_WIDTH, TICK_WIDTH, TICK_LENGTH, LABEL_SIZE


start = 0
end = 8
num = 1000
xs = np.arange(start, end + 1)
alphas = np.linspace(start, end, num)
ys0 = np.cos(2 * np.pi * alphas)
ys1 = np.cos(2 * np.pi * alphas / 1.5)
ys2 = np.cos(2 * np.pi * alphas / np.sqrt(3))

fig, axes = plt.subplots(3, 1, sharex=True)
titles = [r"$R_2=1$", r"$R_2=3/2$", r"$R_2=\sqrt{3}$"]
for ax, ys, title in zip(axes, [ys0, ys1, ys2], titles):
    ax.plot(alphas, ys, lw=LINE_WIDTH)
    ax.plot(xs, [0] * len(xs), marker="o", ls="", ms=MARKER_SIZE)
    ax.set_xlim(start, end)
    ax.set_title(title, fontsize=FONT_SIZE, pad=15)

    for which, spine in ax.spines.items():
        spine.set_linewidth(SPINE_WIDTH)
    ax.tick_params(
        axis="both", which="both", length=TICK_LENGTH, width=TICK_WIDTH,
        labelsize=LABEL_SIZE,
    )

# fig.suptitle("Commensurate/Incommensurate", fontsize=FONT_SIZE)
plt.show()
fig.savefig("demo/ShowCommensurate.jpg", dpi=FIG_DPI)
plt.close("all")
