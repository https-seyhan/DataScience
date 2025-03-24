import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
# Example dataset
df = sm.datasets.get_rdataset("quine", "MASS").data  # count of school absences

# Fit Negative Binomial model
model = smf.glm(formula='Days ~ Age + Sex + Eth + Lrn',
                data=df,
                family=sm.families.NegativeBinomial()).fit()

print(model.summary())

print(np.exp(model.params))
