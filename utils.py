import matplotlib.pyplot as plt

def loss_plot(train_loss, test_loss):
    """Draw test and train loss plot"""
    plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
    plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
    plt.title('Training and Test loss')
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()


def accuracy_plot(train_accuracy, test_accuracy):
    """Draw test and train accuracy plot"""
    plt.plot(range(len(train_accuracy)), train_accuracy, 'b', label='Training Accuracy')
    plt.plot(range(len(train_accuracy)), test_accuracy, 'r', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()
