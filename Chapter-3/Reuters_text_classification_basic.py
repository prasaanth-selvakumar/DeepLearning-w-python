from tensorflow.keras.datasets import reuters
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
try:
    from Common_functions.Text_related.one_hot_encode import text_features_to_vectors, encode_target
    from Common_functions.Train_performance_tracking.learning_curce import  plot_acc_loss_curve
except ModuleNotFoundError:
    import sys
    sys.path.append('./')
    from Common_functions.Text_related.one_hot_encode import text_features_to_vectors, encode_target
    from Common_functions.Train_performance_tracking.learning_curce import plot_acc_loss_curve


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# Train and test data length
print("Training Size", train_data.shape)
print("Test Size", test_data.shape)

train_data_preprocessed = text_features_to_vectors(train_data)
test_data_preprocessed = text_features_to_vectors(test_data)

train_labels_ohe = encode_target(train_labels)
test_labels_ohe = encode_target(test_labels)

model = Sequential([
    Dense(64, activation='relu', input_shape=(10000,)),
    Dense(64, activation='relu'),
    Dense(46, activation='softmax')
])

# Training model on one hot encoded data
model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])

history = model.fit(train_data_preprocessed,
          train_labels_ohe,
          batch_size=512,
          epochs=20,
          validation_split=0.2
          )

print("Model performance using One Hot encoded Data", model.evaluate(test_data_preprocessed, test_labels_ohe))
plot_acc_loss_curve(history.history)  # uses the package to plot learning curves


#  Training Model on Nominal data
model.compile(loss='sparse_categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])

history = model.fit(train_data_preprocessed,
          train_labels,
          batch_size=512,
          epochs=20,
          validation_split=0.2
          )

print("Model Performance Using Nominal Data", model.evaluate(test_data_preprocessed, test_labels))

plot_acc_loss_curve(history.history)  # uses the package to plot learning curves