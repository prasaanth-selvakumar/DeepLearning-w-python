from tensorflow.keras.datasets import imdb
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data.shape, test_data.shape)

#  Checking max word index and max sequence length

print("Max word index",max([max(sequence) for sequence in train_data]))

print("Max len of sequence", max([len(sequence) for sequence in train_data]))

# Decoding sample instances

word_index = imdb.get_word_index()  # this contains words as keys and indices as tokens

reversed_word_index = {value : key for key, value in word_index.items()}

decode_sentences = 5
for i in range(decode_sentences):
    print(" ".join([reversed_word_index.get(j-3,"?") for j in train_data[i]]))


def create_tensors(sequence, dimensions=10000):
    data = np.zeros((sequence.shape[0],dimensions), dtype=np.float32)  # Create a zero tensor with shape
    # - (input_length, dimensions)
    for i in range(sequence.shape[0]):
        data[i, sequence[i]] = 1  # Turning on all words of the sequence
    return data


train_data_preprocessed = create_tensors(train_data)
test_data_preprocessed = create_tensors(test_data)

print("Train Data post Preprocessing: ", train_data_preprocessed[0])
print("Test Data post Preprocessing: ", test_data_preprocessed[0])


model = Sequential([
    Dense(16, activation='relu', input_shape=(10000,)),
    Dense(16,activation='relu'),
    Dense(16,activation='relu'),
    Dense(1,activation='sigmoid')
])  # Model Architecture

train_epochs = 20
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

val_data_preprocessed = train_data_preprocessed[:5000, :]
val_labels = train_labels[:5000]
train_data_preprocessed = train_data_preprocessed[5000:, :]
train_labels = train_labels[5000:]

history = model.fit(train_data_preprocessed, train_labels,
                    validation_data=(val_data_preprocessed,val_labels),
                    epochs=train_epochs,
                    batch_size=512)


history_dict = history.history
epochs_vals = range(train_epochs)


# Plotting the training curves
plt.plot(epochs_vals, history_dict["loss"], 'r', label='training_loss')
plt.plot(epochs_vals, history_dict["val_loss"], 'g', label='validation_loss')
plt.title("Loss Over Epochs")
plt.legend()
plt.show()


plt.plot(epochs_vals, history_dict["accuracy"], 'r', label='training_accuracy')
plt.plot(epochs_vals, history_dict["val_accuracy"], 'g', label='validation_accuracy')
plt.title("Accuracy Over Epochs")
plt.legend()
plt.show()


model.evaluate(test_data_preprocessed,test_labels)

model.predict(test_data_preprocessed)