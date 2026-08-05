from scipy.stats import wasserstein_distance_nd
import numpy as np


def emd(reference, predicted):
    n_events = reference.shape[0]
    distances = np.empty(n_events)
    for event_n in range(n_events):
        u_values = reference[event_n][:, :3]
        v_values = predicted[event_n][:, :3]
        u_weights = reference[event_n][:, 3]
        v_weights = predicted[event_n][:, 3]
        distances[event_n] = wasserstein_distance_nd(
            u_values, v_values, u_weights, v_weights
        )
    return distances



