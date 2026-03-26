import matplotlib.pyplot as plt

def plot_history(history):
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(history.history['loss'],label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Loss Curve")

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'],label='Trainig Accuracy')
    plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
    plt.title("Accuracy Curve")

    plt.show()