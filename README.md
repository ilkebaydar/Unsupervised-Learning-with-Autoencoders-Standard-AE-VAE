# Unsupervised Learning with Autoencoders: Standard AE & VAE

This repository contains implementations of two types of Autoencoders—**Standard Autoencoder (NumPy)** and **Variational Autoencoder (PyTorch)**—applied to the MNIST handwritten digit dataset. This project explores dimensionality reduction, data reconstruction, and the latent space structure of generative models.

## 🚀 Project Overview

The project is divided into two main parts:
1.  **Standard Autoencoder (NumPy):** A manual implementation of forward and backward propagation using only NumPy to understand the mechanics of reconstruction loss (MSE) and bottleneck constraints.
2.  **Variational Autoencoder (VAE) (PyTorch):** A probabilistic generative model implementation using PyTorch, featuring the reparameterization trick and the ELBO (Evidence Lower Bound) loss function.

## 📂 File Structure

* `autoencoder.py`: Contains the `Autoencoder` class implemented in pure NumPy.
* `VAE.py`: Contains the `VAE` class and `vae_loss` function implemented in PyTorch.
* `homework_2.ipynb`: The main notebook where models are trained, visualized, and analyzed.
* `mnist.npz`: The dataset file (auto-downloaded by the notebook).

## 🛠️ Requirements

To run this project, you need the following libraries:
* Python 3.x
* NumPy
* PyTorch
* Matplotlib

You can install the necessary dependencies using:
```bash
pip install torch numpy matplotlib

## 🎓 Academic Note

This project was developed as an assignment for the **BLG454E - Learning From Data** course at **Istanbul Technical University (ITU)**, Faculty of Computer and Informatics.
