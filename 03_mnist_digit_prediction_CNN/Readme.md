# MNIST Handwritten Digit Recognition - README

## 📌 Project Overview
This Jupyter Notebook implements a neural network for recognizing handwritten digits (0-9) from the MNIST dataset using TensorFlow and Keras. The model achieves **92.5% accuracy** and then it is improved to **97.5%** on the test set after training for 5 epochs.

---

## 🏗️ Project Structure

### 1. Data Loading & Exploration
- Loads the MNIST dataset (60,000 training and 10,000 test images)  
- Each image is a 28x28 grayscale pixel array  
- Labels are digits 0-9  

### 2. Data Preprocessing
- **Normalization:** Pixel values scaled from 0-255 to 0-1 range  
- **Flattening:** 28x28 images converted to 784-dimensional vectors  

**Why normalization?** Makes the data easier for the model to learn from

### 3. Model Architecture
Simple neural network with:  
- **Input Layer:** 784 neurons (one for each pixel)  
- **Output Layer:** 10 neurons (one for each digit 0-9)  
- **Activation Function:** Sigmoid  
- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Epochs:** 5 (full passes through the training data)  

### 4. Training & Evaluation
- **Training Accuracy:** ~92.5% after 5 epochs  
- **Test Accuracy:** 92.5% (shows the model generalizes well)   and then improved to 97.5%

**Note:** Too many epochs can cause overfitting

### 5. Prediction & Analysis
- Model outputs probability distribution for each digit  
- `np.argmax()` selects the digit with highest probability  
- Confusion matrix visualizes prediction performance  

---

## 🔧 Key Concepts Explained

**Flattening**  
Converts 2D images (28x28) into 1D arrays (784 elements) so they can be processed by the neural network.

**Epochs**  
- One epoch = one full pass through all training data  
- 5 epochs = model sees each image 5 times  
- Multiple epochs allow gradual learning but risk overfitting  

**Dense Layers**  
All neurons in one layer connect to all neurons in the next layer.

---

## 📊 Performance Metrics
- **Training Loss:** Decreased from 0.47 to 0.27 over 5 epochs  
- **Training Accuracy:** Improved from 87.7% to 92.5%  
- **Test Accuracy:** 92.5% (consistent with training)  

---

## 🚀 How to Run
1. Ensure you have **TensorFlow, Keras, NumPy, Seaborn and Matplotlib** installed  
2. Run cells sequentially in the notebook  
3. The model automatically downloads and processes the MNIST dataset  

---

## 💡 Key Takeaways
- Simple neural networks can achieve high accuracy on MNIST  
- Data preprocessing (normalization, flattening) is crucial  
- The model effectively generalizes from training to test data  
- Confusion matrix helps identify which digits are most commonly confused  

---

## 🛠️ Requirements
- `tensorflow`  
- `keras`  
- `numpy`  
- `matplotlib` 
- `seaborn`  


---



This project demonstrates fundamental deep learning concepts using a classic computer vision dataset, making it an excellent starting point for understanding neural networks and image classification.
