import numpy as np

SEED = 42
SAMPLE_SIZES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 700, 1000, 1300, 1500, 1750, 2000]
NB_SAMPLES = 80
SAMPLE_SIZE = 250
TEST_SIZE = 0.2
KNN_PARAMS = range(1, 21)
LASSO_PARAMS = np.logspace(-6, 0, 20)  
TREE_PARAMS = range(1, 21)

# Enable or disable respective part
KNN = True
LASSO = True
TREE = True