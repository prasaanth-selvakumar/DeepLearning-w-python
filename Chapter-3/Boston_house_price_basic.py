from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np


(train_data, train_target), (test_data, test_traget) = boston_housing.load_data()

# As mentioned in the readme file
# Normalizing the data

normalize_mean = np.mean(train_data)
normalize_std = np.std(train_data)

train_data_normalized = (train_data - normalize_mean) / normalize_std
test_data_normalized = (test_data - normalize_mean) / normalize_std

#  Build Model to use in K-Fold Crossvalidation


def build_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(train_data.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# implementing K-Fold Cross validation using numpy

no_of_folds = 5
indices = np.array(range(len(train_data)))
entries_per_set = len(train_data) // no_of_folds
np.random.shuffle(indices)
epochs_count = 20

performance_val = []
performance_train = []

for i in range(no_of_folds):
    val_indices_start = i * entries_per_set
    val_indices_end = (i+1) * entries_per_set
    val_data = train_data_normalized[val_indices_start:val_indices_end, :]
    val_target = train_target[val_indices_start:val_indices_end]
    if i != 0 and i != (no_of_folds-1):
        partial_train_data = np.vstack([train_data_normalized[0:val_indices_start, :], train_data_normalized[val_indices_end:, :]])
        partial_train_targets = np.vstack([train_target[:val_indices_start].reshape(-1, 1), train_target[val_indices_end:].reshape(-1, 1)])
    elif i == 0:
        partial_train_data = train_data_normalized[val_indices_end:, :]
        partial_train_targets = train_target[val_indices_end:]
    else:
        partial_train_data = train_data_normalized[0:val_indices_start, :]
        partial_train_targets = train_target[0:val_indices_start]
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        epochs=epochs_count, validation_data= (val_data, val_target))
    history_dict = history.history
    performance_val.append(history_dict["val_mae"])
    performance_train.append(history_dict["mae"])

# Average MAE
average_mae_val = [np.mean([j[i] for j in performance_val]) for i in range(epochs_count)]
average_mae = [np.mean([j[i] for j in performance_train]) for i in range(epochs_count)]
plt.plot(range(epochs_count), average_mae, 'r', label='training')
plt.plot(range(epochs_count), average_mae_val, 'r', label='validation')
plt.title("Training vs test Perfromance")
plt.show()
# Post this the model should be trained on the entire data
model = build_model()
model.fit(train_data_normalized, train_target)

print("Model Performance on Test", model.evaluate(test_data_normalized, test_traget))