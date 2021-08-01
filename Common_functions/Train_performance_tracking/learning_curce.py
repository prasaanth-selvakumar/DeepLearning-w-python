import matplotlib.pyplot as plt


def plot_acc_loss_curve(history_dict, accuracy = True, loss=True):
    """
    Plot the learning curves across epochs 
    :param history_dict:
    :param accuracy:
    :param loss:
    :return: Null
    """
    epochs_vals = range(len(history_dict["accuracy"]))
    # Plotting the training curves
    if loss:
        plt.plot(epochs_vals, history_dict["loss"], 'r', label='training_loss')
        plt.plot(epochs_vals, history_dict["val_loss"], 'g', label='validation_loss')
        plt.title("Loss Over Epochs")
        plt.legend()
        plt.show()

    if accuracy:
        plt.plot(epochs_vals, history_dict["accuracy"], 'r', label='training_accuracy')
        plt.plot(epochs_vals, history_dict["val_accuracy"], 'g', label='validation_accuracy')
        plt.title("Accuracy Over Epochs")
        plt.legend()
        plt.show()
