"""
A demonstration of commensurate/incommensurate in 1D for two periodical
system
"""


import matplotlib.pyplot as plt
import numpy as np


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
    ax.plot(alphas, ys)
    ax.plot(xs, [0] * len(xs), marker="o", ls="")
    ax.set_xlim(start, end)
    ax.set_title(title)

plt.show()
fig.savefig("demo/ShowCommensurate.jpg")
plt.close("all")
