import matplotlib.pyplot as plt
import os


def loss_plot(train_loss, test_loss, plot_path):
    """Draw test and train loss plot"""
    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
    plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
    plt.title('Training and Test loss')
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.savefig(str(plot_path))


def accuracy_plot(train_accuracy, test_accuracy, plot_path):
    """Draw test and train accuracy plot"""
    plt.figure()
    plt.plot(range(len(train_accuracy)), train_accuracy, 'b', label='Training Accuracy')
    plt.plot(range(len(train_accuracy)), test_accuracy, 'r', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend()
    plt.savefig(str(plot_path))


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
