def run_evaluation(model=None, x_test=None, y_test=None):
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import config

    if model is None:
        model = tf.keras.models.load_model(config.MODEL_PATH)

    if x_test is None or y_test is None:
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_test = x_test.astype("float32") / 255.0
        y_test = tf.keras.utils.to_categorical(y_test, config.NUM_CLASSES)

    # Load trained model
    model = tf.keras.models.load_model(config.MODEL_PATH)

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)

    # Predictions
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap='Blues')

    plt.title("Confusion Matrix")
    plt.show()
