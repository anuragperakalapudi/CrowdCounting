# Depth-Embedded Lightweight Crowd Counting

This repo contains a **practical crowd counting model** built for situations where you want **reasonable accuracy without heavy compute**. The core idea is simple: take a small CNN that can run fast, and make it smarter by giving it **depth-aware features**.

Instead of chasing state-of-the-art accuracy with massive models, this project focuses on the tradeoff between:
- speed
- model size
- accuracy

The result is a depth-embedded, lightweight crowd density estimator that outperforms basic lightweight models while staying usable in near real-time settings.

---

## What’s in this repo

- `Crowd_Counting_Model.ipynb`  
  End-to-end implementation: preprocessing, density map generation, model definition, training, and evaluation.

- `Real-Time Crowd Counting with Depth-Embedded Lightweight Neural Networks.pdf`  
  A deeper technical write-up for anyone who wants full details.

The notebook is the main artifact. The paper is optional reading.

---

## How the approach works

Most lightweight crowd counting models only see a flat image. That makes it hard to reason about scale and distance, especially in dense scenes.

This project adds a **depth embedding module** that learns spatial distance information and fuses it with image features before predicting a density map.

High level flow:
1. Input image + head annotations
2. Generate a ground-truth density map using a Gaussian kernel
3. Extract image features with a small CNN
4. Extract depth features with a separate CNN
5. Combine both feature streams
6. Predict a crowd density map
7. Sum the density map to get the final count

---

## Model & Algorithms

- Convolutional Neural Networks (CNNs)
- Density map regression (instead of object detection)
- Gaussian-based density map generation
- Feature fusion (RGB + depth embeddings)
- Mean Absolute Error (MAE) for evaluation

The model is inspired by LCDNet-style architectures but extended with learned depth features.

---

## Frameworks & Libraries

- **PyTorch** – model definition and training  
- **NumPy** – numerical operations  
- **OpenCV** – image loading and preprocessing  
- **SciPy** – Gaussian filtering and `.mat` file handling  
- **Matplotlib** – visualization  
- **Jupyter Notebook** – experimentation and analysis  

Training and experiments were run in Google Colab.

---

## Dataset setup

- Images paired with head annotations (`.mat` files)
- Head locations converted into density maps
- Fixed Gaussian kernel used for efficiency

**Split**
- 300 training images  
- 50 test images  

---

## Results (high level)

The depth-embedded lightweight model:
- Performs significantly better than lightweight models without depth
- Is much faster than full depth-aware crowd counting networks
- Works best in very dense crowd scenes

It struggles more when large non-human objects (trees, buildings, cars) dominate the frame, which is a known limitation of density-based methods.

---

## Why this is interesting

- Shows how **small architectural changes** can meaningfully improve lightweight models
- Explores the accuracy vs. speed tradeoff instead of just chasing benchmarks
- Designed with **real-time feasibility** in mind
- Good example of applied computer vision beyond toy datasets

---

## Possible improvements

- Adaptive Gaussian kernel sizing
- Better background-object suppression
- More robust depth representations
- Extension to video-based crowd counting

---

## Authors

Anurag Perakalapudi  
Aaryan Sumesh
