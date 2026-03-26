import tensorflow as tf
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index = None):
    
    grad_model= tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.outputs[0]]
    )

    with tf.GradientTape() as tape:
        #forward pass
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:,pred_index]#extract that class model
    grads = tape.gradient(class_channel, conv_outputs)#compute gradients. Key step answering "Which prts of the feature maps most affect this class?"
    pooled_grads = tf.reduce_mean(grads,axis=(0,1,2)) # average the gradients . we collapse the spatial dimetnsion. Each channel gets a single importance score
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[...,tf.newaxis] # heatmap = weighted sum fo feature maps. So important feature are amplified and unimportant one is supperssed.
    heatmap = tf.squeeze(heatmap) #
    heatmap = tf.maximum(heatmap,0)/tf.reduce_max(heatmap)# remove the negative values nad normalize. values in [0,1]
    return heatmap.numpy() # give a 2D map