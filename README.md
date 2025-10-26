# VGG-6 CIFAR-10 Classification Project

A PyTorch implementation of a VGG-6 convolutional neural network for CIFAR-10 image classification with data augmentation and batch normalization.

## 📋 Project Overview

This project implements a simplified VGG architecture (VGG-6) to classify images from the CIFAR-10 dataset. The model achieves **80.63% test accuracy** after 20 epochs of training.

### Model Architecture
- **6 Convolutional Layers**: 64-64-128-128 filters
- **Batch Normalization**: Applied after each convolution
- **Activation Function**: ReLU
- **Pooling**: 2 MaxPool2d layers
- **Classifier**: Single fully connected layer (128 → 10 classes)

### Dataset
- **CIFAR-10**: 60,000 32x32 color images in 10 classes
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Training Set**: 50,000 images
- **Test Set**: 10,000 images

## 🚀 Quick Start

### 1. Setup Environment

#### Prerequisites
- Python 3.7+
- pip package manager

#### Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python DLassign1.py
```

### 3. Test the Model
```bash
python test.py
```

## 🛠️ Detailed Setup Instructions

### Step 1: Clone/Download Project
Download or clone this project to your local machine:
```bash
# If using git
git clone <your-repository-url>
cd DLAssignment1

# Or download and extract the ZIP file
```

### Step 2: Install Dependencies
The project requires the following packages:
- **torch** (≥2.0.0): PyTorch deep learning framework
- **torchvision** (≥0.15.0): Computer vision utilities
- **Pillow** (≥9.0.0): Image processing
- **numpy** (≥1.21.0): Scientific computing

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
Test if PyTorch is installed correctly:
```python
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## 🎯 Training Instructions

### Basic Training
Run the training script with default parameters:
```bash
python DLassign1.py
```

### Training Configuration
The model is configured with the following parameters:
- **Epochs**: 20
- **Batch Size**: 64
- **Learning Rate**: 0.01
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Device**: Automatically detects CUDA/CPU

### Training Process
1. **Data Loading**: CIFAR-10 dataset is automatically downloaded to `./data/`
2. **Data Augmentation**: 
   - Random cropping (32×32 with padding=4)
   - Random horizontal flipping
   - Cutout regularization (16×16 holes)
   - Normalization with CIFAR-10 statistics
3. **Training Loop**: 20 epochs with real-time accuracy monitoring
4. **Model Saving**: Trained model saved as `trained_vgg6_model.pth`

### Expected Training Output
```
Starting training with parameters:
  - Epochs: 20
  - Batch Size: 64
  - Learning Rate: 0.01
  - Optimizer: Adam
  - Activation: ReLU (built into VGG architecture)
  - Device: cpu/cuda
--------------------------------------------------
Epoch 0 - Train_Loss: 1.6217, Train_acc: 43.40, Test_acc: 45.82
Epoch 1 - Train_Loss: 1.5315, Train_acc: 55.93, Test_acc: 58.74
...
Epoch 19 - Train_Loss: 0.6335, Train_acc: 78.70, Test_acc: 80.63
Model saved to trained_vgg6_model.pth
Training completed! Run 'python test.py' to evaluate the model on test data.
```

## 🧪 Testing Instructions

### Basic Testing
Evaluate the trained model on the test dataset:
```bash
python test.py
```

### What the Test Script Does
1. **Loads Trained Model**: Automatically loads `trained_vgg6_model.pth`
2. **Loads Test Data**: CIFAR-10 test set (10,000 images)
3. **Evaluates Performance**: 
   - Overall test accuracy
   - Per-class accuracy breakdown
   - Detailed progress tracking

### Expected Test Output
```
Using device: cpu
Model loaded successfully from trained_vgg6_model.pth
Model was trained for 20 epochs
Evaluating model on test dataset...
Processed 0/10000 samples
Processed 3200/10000 samples
Processed 6400/10000 samples
Processed 9600/10000 samples

============================================================
OVERALL TEST ACCURACY: 80.63%
============================================================

Per-class Accuracy:
------------------------------
airplane    :  86.30% (863/1000)
automobile  :  85.30% (853/1000)
bird        :  72.30% (723/1000)
cat         :  69.10% (691/1000)
deer        :  72.60% (726/1000)
dog         :  78.70% (787/1000)
frog        :  82.60% (826/1000)
horse       :  82.60% (826/1000)
ship        :  84.00% (840/1000)
truck       :  92.80% (928/1000)
------------------------------

Final Result: Test Accuracy = 80.63%
```

## 📁 Project Structure

```
DLAssignment1/
│
├── DLassign1.py           # Main training script
├── test.py               # Model evaluation script
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── trained_vgg6_model.pth  # Saved model (created after training)
│
└── data/                # Dataset directory (auto-created)
    └── cifar-10-batches-py/
        ├── batches.meta
        ├── data_batch_1
        ├── data_batch_2
        ├── data_batch_3
        ├── data_batch_4
        ├── data_batch_5
        ├── test_batch
        └── readme.html
```

## 🔧 Customization

### Modify Training Parameters
Edit the parameters in `DLassign1.py`:
```python
# Training parameters
batch_size = 64          # Batch size
learning_rate = 0.01     # Learning rate
epochs = 20              # Number of epochs
optimizer = optim.Adam   # Optimizer type
```

### Model Architecture
Modify the VGG configuration:
```python
# VGG-6 configuration
cfg_vgg6 = [64, 64, 'M', 128, 128, 'M']
# Add more layers: cfg_vgg8 = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M']
```

## 🚨 Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError: No module named 'torch'
```bash
pip install torch torchvision
```

#### 2. CUDA out of memory
- Reduce batch size in `GetCifar10()` function
- Use CPU by setting: `device = torch.device('cpu')`

#### 3. Model file not found
- Ensure you've run `python DLassign1.py` first to train and save the model
- Check that `trained_vgg6_model.pth` exists in the project directory

#### 4. Slow training on CPU
- Consider using GPU if available
- Reduce batch size or number of epochs for faster training

### Performance Tips
- **GPU Acceleration**: Install CUDA-compatible PyTorch for faster training
- **Batch Size**: Increase if you have more memory, decrease if you get memory errors
- **Workers**: Increase `num_workers` in DataLoader for faster data loading

## 📊 Results

### Model Performance
- **Test Accuracy**: 80.63%
- **Training Time**: ~20 minutes on CPU (depends on hardware)
- **Model Size**: ~1.2 MB

### Best Performing Classes
1. Truck: 92.80%
2. Airplane: 86.30%
3. Automobile: 85.30%
4. Ship: 84.00%

## 📝 License

This project is for educational purposes. Feel free to modify and distribute.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Happy Training! 🎉**