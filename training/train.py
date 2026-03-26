import config

def train_model(model, train_ds,val_ds,callbacks):
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks
    )
    return history