def run_training():
    from data.loader import load_cifar10
    from data.pipeline import create_datasets
    from models.resnet import build_resnet
    from training.callbacks import get_callbacks
    from training.train import train_model
    from evaluation.plots import plot_history
    from gradCAM import make_gradcam_heatmap
    import config 

    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    #load data
    (x_train, y_train),(x_test,y_test)= load_cifar10()

    class_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    #create datasets
    train_ds,val_ds = create_datasets(x_train,y_train,x_test,y_test)

    #build model
    model = build_resnet()



    steps_per_epoch = len(x_train) // 128

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate= config.INITIAL_LEARNING_RATE,
    decay_steps=steps_per_epoch * config.EPOCHS,
    alpha=config.S_ALHPA
)
  
    optimizer = tf.keras.optimizers.SGD(
        learning_rate = lr_schedule,
        momentum = config.MOMENTUM,
        nesterov = True
    )

    model.compile(
        optimizer = optimizer,
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics = ['accuracy']
    )


    #train
    history = train_model(model, train_ds,val_ds,get_callbacks())


    #evaluate the model first
    test_loss,test_acc = model.evaluate(x_test,y_test)
    print( "Test Loss:", test_loss, "and Test Accuracy:", test_acc)


    #Plot training curves. 
    plot_history(history)


    #model.predict(x_test[:1])
    _ = model([x_test[:1]])
    #tp model.input and output ready for keras

    #genrate heatmap
    img = x_test[0]
    img_array = np.expand_dims(img,axis=0)
    heatmap = make_gradcam_heatmap(img_array, model, "last_conv")
    heatmap= tf.image.resize(heatmap[...,tf.newaxis],(32,32),method="bilinear")
    heatmap = tf.squeeze(heatmap).numpy()

    #visulize the heatmap
    plt.figure(figsize=(12,4))
    #heatmap
    plt.subplot(1,2,1)
    im=plt.imshow(heatmap,cmap="jet")
    plt.title("Grad-CAM Heatmap")
    plt.colorbar(im, fraction=0.046, pad= 0.04)
    plt.axis("off")

    #overlay heatmap on original image(best visualization)
    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.imshow(heatmap, cmap="jet", alpha=0.4)
    pred = np.argmax(model.predict(x_test[:1]),axis=1)[0]
    plt.title(f"Model Attention (Pred:{pred})")
    plt.axis("off")
    plt.show()

    return model, (x_test,y_test)
