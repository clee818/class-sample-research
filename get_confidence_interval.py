import numpy as np


def get_confidence_interval(y_pred, y_true, verbose=True):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
        
        score = sum(y_pred[indices]==y_true[indices])/len(y_pred)
        bootstrapped_scores.append(score)


    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    accuracy = sum(y_pred==y_true)/len(y_pred)
    

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    if verbose:
        print("Accuracy: {} [{:0.3f} - {:0.3}]".format(
          accuracy, confidence_lower, confidence_upper))
    return confidence_lower, confidence_upper, accuracy