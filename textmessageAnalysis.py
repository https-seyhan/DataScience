#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 22:19:49 2019

@author: saul
"""
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pymc3 as pm


os.chdir("/home/saul/pythonWork")

#textfile = pd.read_csv("txtdata.csv", sep=",", header=None)
textfile = np.loadtxt("txtdata.csv")

#print(textfile.describe())



size = len(textfile)
print(size)
alpha = 1 / textfile.mean()

plt.bar(np.arange(size), textfile, color="#348ABD")
plt.xlabel("Time (days)")
plt.ylabel("Text message received")
plt.title("Did the user's texting habit change over time?")
plt.xlim(0, size)

with pm.Model() as model:
    
    lambda_1 = pm.Exponential('lambda_1', alpha) # create stochastic variable
    lambda_2 = pm.Exponential('lambda_2', alpha) #create stochastic variable
    print(lambda_1)
    tau = pm.DiscreteUniform("tau", lower=0, upper=size)
    print("Random output:", tau.random(), tau.random(), tau.random())

@pm.deterministic
