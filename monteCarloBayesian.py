#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: saul
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pymc3 as pm # PyMC3 is a Python package for Bayesian statistical modeling 


os.chdir("/home/saul/pythonWork")

#textfile = pd.read_csv("txtdata.csv", sep=",", header=None)
textfile = np.loadtxt("txtdata.csv")

#print(textfile.describe())
#@pm.deterministic

size = len(textfile)
print(size)
#alpha = 1 / size
alpha = 1.0 / textfile.mean()


with pm.Model() as model:
    lambda_1 = pm.Exponential('lambda_1', alpha) # lambda in poisson distribution
    lambda_2 = pm.Exponential('lambda_2', alpha)
    tau = pm.DiscreteUniform("tau", lower=0, upper=size)

with pm.Model():
    x = pm.Normal('x', mu=0, sd=1)
    plus_2 = pm.Deterministic('x plus 2', x + 2)
    print("X: ", plus_2)

def lambda_ (tau=tau, lambda_1 = lambda_1, lambda_2 = lambda_2):
    out = np.zeros(size)
    out[:tau] = lambda_1
    out[tau:] = lambda_2
    
print(lambda_(tau.random(), lambda_1.random(), lambda_2.random()))


    

