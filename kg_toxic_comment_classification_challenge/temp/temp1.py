#!/usr/bin/python

from sklearn.preprocessing import PolynomialFeatures

import numpy as np


X = np.arange(6).reshape(3,2)

X


poly = PolynomialFeatures(degree=3)

poly.fit_transform(X)
