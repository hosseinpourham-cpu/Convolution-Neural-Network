import tensorlflow as tf
import config

def load_cifar10():
    (x_train, y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
    #normalize and cast images
    x_train = x_train.astype("float32")/255.0
    x_test = x_test.astype("float32")/255.0

    y_train = tf.keras.utils.to_categorical(y_train,config.NUM_CLASSES).astype("float32")
    y_test = tf.keras.utils.to_categorical(y_test,config.NUM_CLASSES).astype("float32")
    return (x_train, y_train),(x_test,y_test)