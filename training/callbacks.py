import tensorflow as tf
import config

def get_callbacks():
    return[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            config.MODEL_PATH, #"output'models'best_model.keras"
            monitor='val_loss',
            save_best_only= True,
            verbose=1
        )
       
    ]
 
   