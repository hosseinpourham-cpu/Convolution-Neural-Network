#CIFAR-10 Image Classification with ResNet and Grad-CAM
This project implements a convolutional Neural Network (ResNet-style) for image classification on the CIFAR-10 datasets using TensorFlow/keras.

It contains :
- Data augmentation
- Resifual blocks (ResNet architecture)
- MixUp regularization : the current version is commented. Suggestion: use beta distribution with random beta. good for general datasets
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

- MixUp data augmentation
- Cosine learning rate schedule
- EarlyStopping
- ModelCheckpoint (best model saved)

---

## Results 

- Test Accuracy : ~ %
- Validation Accuracy : ~ %

---

## Grad

Generates heatmaps to highlight which parts of the image the model focuses on.

Example outputs : 
- Heatmap
- Overlay on original image

---

## How to Run

```bash
pip install -r requirements.txt
python train.py
```

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


## Author

Hamed Hosseinpour