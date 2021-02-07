#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: saul
"""
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# PyMC3 is a Python package for Bayesian statistical modeling and probabilistic machine learning 
# which focuses on advanced Markov chain Monte Carlo and variational fitting algorithms
import pymc3 as pm 

os.chdir("/home/saul/pythonWork")

#textfile = pd.read_csv("txtdata.csv", sep=",", header=None)
textfile = np.loadtxt("txtdata.csv")

#print(textfile.describe())

size = len(textfile)
print(size)
alpha = 1 / textfile.mean()

# plot bar graph
plt.bar(np.arange(size), textfile, color="#348ABD")
plt.xlabel("Time (days)")
plt.ylabel("Text message received")
plt.title("Did the user's texting habit change over time?")
plt.xlim(0, size)

# Create the MCMC model with its parameters (lambdas, tau)
with pm.Model() as model:
    lambda_1 = pm.Exponential('lambda_1', alpha) # create stochastic variable
    lambda_2 = pm.Exponential('lambda_2', alpha) #create stochastic variable
    print(lambda_1)
    tau = pm.DiscreteUniform("tau", lower=0, upper=size) # tau has Uniform distribution
    print("Random output:", tau.random(), tau.random(), tau.random())
    

n_data_points = size   
idx = np.arange(n_data_points)
with model:
    lambda_ = pm.math.switch(tau >= idx, lambda_1, lambda_2)            

#pm.Poisson("obs", lambda_, value = textfile, observed=True)

with model:
    obs = pm.Poisson("obs", lambda_, observed=textfile)
print(obs.tag.test_value)

model = pm.Model([obs, lambda_1, lambda_2, tau])
print(model)

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000)
