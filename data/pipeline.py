import tensorflow as tf
import config

def mixup(x,y , alpha):
    #beta distribution
    lam1 = tf.random.gamma([],alpha,1.0) 
    lam2 = tf.random.gamma([],alpha,1.0)
    lam = lam1/(lam1+lam2)
 
    batch_size = tf.shape(x)[0]
    index = tf.random.shuffle(tf.range(batch_size))
    
    x2 = tf.gather(x,index)
    y2 = tf.gather(y,index)

    mixed_x= lam*x  + (1-lam)* x2 
    mixed_y = lam *y + (1-lam)* y2
    return mixed_x, mixed_y

def create_datasets(x_train,y_train, x_test,y_test, batch_size=config.BATCH_SIZE):
    train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train))
   # train_ds = (train_ds.shuffle(50000).batch(batch_size).map(mixup,num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE))
    train_ds = (train_ds.shuffle(50000).batch(batch_size).prefetch(tf.data.AUTOTUNE))

    val_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds