#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: saul
"""
import scipy.stats as stats
import numpy as np
from matplotlib import pyplot as plt
a = np.arange(16)
print(type(a))
poi = stats.poisson

lambda_ = [1.5, 4.25]
colors = ["#348ABD","#A60628"]

print(colors)
plt.bar(a, poi.pmf(a, lambda_[0]), color=colors[0], label="$\lambda_ = %.1f$" % lambda_[0], alpha = 0.60, 
        edgecolor= colors[0], lw="3")

plt.bar(a, poi.pmf(a, lambda_[1]), color=colors[1], label="$\lambda_ = %.1f$" % lambda_[1], alpha = 0.60, 
        edgecolor= colors[0], lw="3")
plt.xticks(a + 0.4, a)
plt.legend()
plt.ylabel("Probability of $k$")
plt.xlabel("$k$")
plt.title("Probability mass function of a Possion random variable, \ deiffering \$\lambda$ values")
