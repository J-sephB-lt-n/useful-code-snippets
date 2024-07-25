"""
TAGS: data|density|dist|distfit|distribution|fit|stats|statistics|univariate
DESCRIPTION: Find the best-fitting distribution for univariate continuous data using python `distfit` package
REQUIREMENTS: pip install distfit # https://github.com/erdogant/distfit
"""

import random

import matplotlib.pyplot as plt
import numpy as np
from distfit import distfit

# simulate data #
x = []
for _ in range(1_000):
    if random.uniform(0, 1) < 0.3:
        x.append(random.gauss(mu=0, sigma=5))
    else:
        x.append(random.gauss(mu=40, sigma=15))

plt.figure(figsize=(8, 5))
plt.hist(x, bins=30)
plt.show()

dfit = distfit()  # Initialise distribution fitter
dfit.fit_transform(np.array(x))  # Fit distributions on empirical data X
dfit.plot()  # show best-fitting distribution found
