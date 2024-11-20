import numpy as np
from sklearn.model_selection import train_test_split

from constants import *

def prepare_learning_samples(x, y, nb_samples=NB_SAMPLES, sample_size=SAMPLE_SIZE, test_size=TEST_SIZE):
    """
    Split the data into a list of learning sample (input, output), inputs of test sample and its outputs for verification. 
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=SEED)
    # Use a seeded random generator
    rng = np.random.default_rng(SEED)
    learning_samples = []
    for _ in range(nb_samples):
        idxs = rng.choice(len(x_train), size=sample_size, replace=False)
        learning_samples.append((x_train[idxs], y_train[idxs]))
    
    return learning_samples, x_test, y_test

def train_and_predict(learning_samples, x_test, model):
    """
    Train models on each learning sample and generate predictions on the test set.
    """
    predictions = []
    
    for x_train, y_train in learning_samples:
        model.fit(x_train, y_train)
        
        preds = model.predict(x_test)
        predictions.append(preds)
    
    return np.array(predictions)

def compute_metrics(predictions, true_values):
    """
    Compute some metrics based on the prediction and the true_values.
    """
    total_error = np.mean((predictions - true_values) ** 2)
    mean_prediction = np.mean(predictions, axis=0)
    variance = np.var(predictions, axis=0)
    bias_squared = np.mean((mean_prediction - true_values) ** 2)
    
    return total_error, np.mean(variance), bias_squared
