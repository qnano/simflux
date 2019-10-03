# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np

from smlmlib.gaussian import Gaussian
from smlmlib.base import SMLM

smlm = SMLM(debugMode=False)


if __name__ == "__main__":
    g = Gaussian(smlm)

    mu, fi = g.ComputeFisherMatrix([5, 5, 1000, 1], sigma=2, imgw=12)
    smp = np.random.poisson(mu)
    plt.figure()
    plt.imshow(smp[0])

    logl = np.sum(smp * np.log(mu) - mu)

    print(logl)
