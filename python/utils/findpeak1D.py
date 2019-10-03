# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Curve-fit y around the peak to find subpixel peak x
def quadraticpeak(y, x=None, npts=7, plotTitle=None):
    if x is None:
        x = np.arange(len(y))
    xmax = np.argmax(y)
    W = int((npts + 1) / 2)
    window = np.arange(xmax - W + 1, xmax + W)
    window = np.clip(window, 0, len(x) - 1)
    coeff = np.polyfit(x[window], y[window], 2)

    if plotTitle:
        plt.figure()
        plt.plot(x[window], y[window], label="fit")
        sx = np.linspace(x[xmax - W], x[xmax + W], 100)
        plt.plot(sx, np.polyval(coeff, sx), label="fit")
        plt.legend()
        plt.title(plotTitle)

    return -coeff[1] / (2 * coeff[0])


if __name__ == "__main__":

    x = np.arange(100)
    max = 50
    y = -(x - max) ** 2 + np.random.uniform(-5, 5, size=len(x))

    plt.figure()
    plt.plot(x, y)
    print(quadraticpeak(y, x, npts=12, plotTitle="Fit"))
