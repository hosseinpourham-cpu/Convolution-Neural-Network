#CIFAR-10 Image Classification with ResNet and Grad-CAM
This project implements a convolutional Neural Network (ResNet-style) for image classification on the CIFAR-10 datasets using TensorFlow/keras.

It contains :
- Data augmentation
- Resifual blocks (costum ResNet architecture)
- MixUp regularization
- Learning rate scheduling
- Early stopping and checkpointing
- Evaluation with confusion matrix
- Grad-CAM visualization for model interpretability.

---

## Dataset 

- CIFAR-10 (60k images, 10 classes)
- Images are 32x32 RGB

---

## Model
- Custom lightweight ResNet
- Batch Normalization + Dropout 
- L2 regularization
- Glocal Average Pooling 

---

## Training Features 

- Cosine learning rate schedule
- SGD optmizer
- EarlyStopping
- ModelCheckpoint (best model saved)

---

## Results 

- Training Accuracy =  0.9999
- Validation Accuracy = 0.9368
 (Test Accuracy = 0.9358999729156494)
---

## Grad

Generates heatmaps to highlight which parts of the image the model focuses on.

Example outputs : 
- Heatmap
- Overlay on original image

---


## Setup
python -m venv tf_env
pip install -r requirements.txt


---

## Project Structure 

```
project/
│
├── data/ loader.py, pipeline.py
├── evalualtion/ evaluate.py, plots.py
├── explainability/gradCAM.py
├── models/ resnet.py
├── outputs/ heatmaps, Plots
├── training/ callbakcs.py, train.py, train_module.py
│
├── config.py
├── main.py
│
├── README.md
├── requirements.txt

```

---

## Future Improvements 

- Try deeper ResNet variants (Net32 for instance, longer runtime. Check the model folder.)
- Add CutMix augmnetation (longer runtime, useful for high accuracy. Check the data folder.)
- Train on GPU (WSL2 or Linux)
- Add MixUp/MixCut data augmentation and smoothen some parameters to have a tradeOff between Training_accuracy and Validation_accuracy. Here is an example: 
- - Add in data augmentation block ( in resnet.py):  x = layers.RandomRotation(0.05)(x)
- - Set : Res_DROPOUT_RATE = 0.3, L2_WEIGHT = 1e-3, INITIAL_LEARNING_RATE = 0.02, EARLY_STOP_PATIENCE = 15
- - Add mixUp in pipeline.
- - This gives Trainng_accuracy= 0.9640, Validation_accuracy= 0.9420 with Test Accuracy: 0.9419999718666077


## Author

Hamed Hosseinpour