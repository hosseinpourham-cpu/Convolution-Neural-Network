from tensorflow.keras import layers, models
import tensorflow as tf
import config

def res_block(x, filters, downsample =False, name=None):
    shortcut = x

    stride=2 if downsample else 1

    #first conv
    x=layers.Conv2D(filters, (3,3), strides= stride, padding="same",
                    kernel_regularizer= tf.keras.regularizers.l2(config.L2_WEIGHT))(x)
    x= layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    #second conv
    x= layers.Conv2D(filters, (3,3), padding="same",
                     name=name,
                     kernel_regularizer=tf.keras.regularizers.l2(config.L2_WEIGHT))(x)
    x= layers.BatchNormalization()(x)

    #adjust shortcut if needed
    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters,(1,1),strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    #add skip connection
    x=layers.Add()([x,shortcut])
    return layers.ReLU()(x)   


def build_resnet():
    inputs=layers.Input(shape=config.INPUT_SHAPE) #(32,32,3)
    #data augmentation
    x=layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomCrop(32, 32)(layers.ZeroPadding2D(4)(x))
    x = layers.RandomBrightness(0.05)(x)

    #initial Conv
    x=layers.Conv2D(32,(3,3),padding="same")(x)
    x=layers.BatchNormalization()(x)
    x=layers.ReLU()(x)
    #Redisual blocks
    x=res_block(x,32)
    x=res_block(x,32)
    x=res_block(x,32)
    x=res_block(x,64,downsample=True)
    x=res_block(x,64)
    x=res_block(x,128,downsample=True)
    x=res_block(x,128)
    x=res_block(x,128, name="last_conv")
    #Classifier
    x= layers.GlobalAveragePooling2D()(x)
    #lighter head
    x=layers.Dense(128,activation='relu')(x)
    x=layers.Dropout(config.Res_DROPOUT_RATE)(x)

    output = layers.Dense(10,activation='softmax')(x)
    return models.Model(inputs,output)