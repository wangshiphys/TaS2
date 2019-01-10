"""
An illustration of commensurate/incommensurate in 1D
"""


import matplotlib.pyplot as plt
import numpy as np


line_width = 5
marker_size = 10

start = 0
end = 8
num = 1000
xs = np.arange(start, end + 1)
alphas = np.linspace(start, end, num)
ys0 = np.cos(2 * np.pi * alphas)
ys1 = np.cos(2 * np.pi * alphas / 1.5)
ys2 = np.cos(2 * np.pi * alphas / np.sqrt(3))

fig, axes = plt.subplots(3, 1, sharex=True)
titles = [r"$R_2=1$", r"$R_2=\frac{3}{2}$", r"$R_2=\sqrt{3}$"]
for ax, ys, title in zip(axes, [ys0, ys1, ys2], titles):
    ax.plot(alphas, ys, lw=line_width)
    ax.plot(xs, [0]*len(xs), marker="o", ls="", ms=marker_size)
    ax.set_xlim(start, end)
    ax.set_title(title, fontsize="x-large")
fig.suptitle("Commensurate/Incommensurate", fontsize="xx-large")
plt.show()
fig.savefig("ShowCommensurate.jpg", dpi=1000)
plt.close("all")
