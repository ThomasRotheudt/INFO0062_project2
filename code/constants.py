import numpy as np

SEED = 42
SAMPLE_SIZES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 700, 1000, 1300, 1500, 1750, 2000]
KNN_PARAMS =  range(1, 200, 5)
LASSO_PARAMS = np.logspace(-6, 0, 20)  
TREE_PARAMS = range(1, 21)
NB_ESTIMATOR = range(1, 10+1)
BASE_DEPTH = 5
TREE_PARAMS_Q2_5 = range(1, 11, 2)

# Enable or disable respective part
KNN = True
LASSO = True
TREE = True
# For Q2.5
CHANGE_DEPTH = True
BAGGING = True
BOOSTING = True