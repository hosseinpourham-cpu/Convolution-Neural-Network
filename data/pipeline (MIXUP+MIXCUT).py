import tensorflow as tf
import config

def mixup(x,y , alpha=config.ALPHA):
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

def cutmix(x, y, alpha=config.ALPHA):
    batch_size = tf.shape(x)[0]
    indices = tf.random.shuffle(tf.range(batch_size))

    x2 = tf.gather(x, indices)
    y2 = tf.gather(y, indices)

    lam1 = tf.random.gamma([], alpha, 1.0)
    lam2 = tf.random.gamma([], alpha, 1.0)
    lam = lam1 / (lam1 + lam2)

    h = tf.shape(x)[1]
    w = tf.shape(x)[2]

    cut_rat = tf.sqrt(1.0 - lam)
    cut_w = tf.cast(w * cut_rat, tf.int32)
    cut_h = tf.cast(h * cut_rat, tf.int32)

    cx = tf.random.uniform([], 0, w, dtype=tf.int32)
    cy = tf.random.uniform([], 0, h, dtype=tf.int32)

    x1 = tf.clip_by_value(cx - cut_w // 2, 0, w)
    x2b = tf.clip_by_value(cx + cut_w // 2, 0, w)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, h)
    y2b = tf.clip_by_value(cy + cut_h // 2, 0, h)

    # build mask
    mask = tf.ones_like(x)
    mask = tf.tensor_scatter_nd_update(
        mask,
        indices=tf.stack(tf.meshgrid(
            tf.range(batch_size),
            tf.range(y1, y2b),
            tf.range(x1, x2b),
            indexing='ij'
        ), axis=-1),
        updates=tf.zeros([batch_size, y2b - y1, x2b - x1, 3])
    )

    mixed_x = x * mask + x2 * (1 - mask)

    lam_adjusted = 1 - (
        tf.cast((x2b - x1) * (y2b - y1), tf.float32) /
        tf.cast(h * w, tf.float32)
    )

    mixed_y = lam_adjusted * y + (1 - lam_adjusted) * y2

    return mixed_x, mixed_y

def aug(x, y):
    return tf.cond(
        tf.random.uniform([]) < 0.5,
        lambda: mixup(x, y),
        lambda: cutmix(x, y)
    )


def create_datasets(x_train,y_train, x_test,y_test, batch_size=config.BATCH_SIZE):
    train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    train_ds = (train_ds.shuffle(50000).batch(batch_size).map(aug,num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE))

    val_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds