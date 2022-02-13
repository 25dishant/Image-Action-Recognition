import matplotlib.pyplot as plt
import numpy as np
import os

def plot_history(train_hist, val_hist, y_label, filename, labels=["train", "validation"]):
    """
    Plot training and validation history
    
    Args:
        train_hist: numpy array consisting of train history values (loss/ accuracy metrics)
        valid_hist: numpy array consisting of validation history values (loss/ accuracy metrics)
        y_label: label for y_axis
        filename: filename to store the resulting plot
        labels: legend for the plot
        
    Returns:
        None
    """
    # Plot loss and accuracy
    xi = [i for i in range(0, len(train_hist))]
    plt.plot(train_hist, label = labels[0])
    plt.plot(val_hist, label = labels[1])
    plt.xticks(xi)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.savefig(filename)
    plt.show()
    # plt.close()

num = 1
model_dir = os.path.join(os.path.expanduser("~"), "Delete after Test5")
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)


# train_hist = np.random.random((10))
train_hist = [2.399728775024414, 2.3725202083587646, 2.3626351356506348, 2.349236488342285, 2.337948799133301, 2.3112998008728027, 2.2819700241088867, 2.2487759590148926, 2.218759298324585, 2.1831212043762207]

val_hist = np.random.random((10))
print(train_hist)
print(val_hist)
plot_history(train_hist,val_hist,"Loss",os.path.join(model_dir, "loss-{}".format(num)),labels=["train","validation"])
plot_history(train_hist,val_hist,"Loss",os.path.join(model_dir, "loss-{}".format(num)),labels=["train","validation"])