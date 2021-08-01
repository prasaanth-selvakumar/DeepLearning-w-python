import numpy as np


def text_features_to_vectors(sequences, dimensions = 10000):
    data = np.zeros((sequences.shape[0],dimensions), dtype=np.float32)
    for i in range(sequences.shape[0]):
        data[i, sequences[i]] = 1
    return data


def encode_target(vector):
    max_in_seq = vector.max() +1
    data = np.zeros((vector.shape[0], max_in_seq), dtype=np.float32)
    for i in range(vector.shape[0]):
        data[i, vector[i]] = 1
    return data

