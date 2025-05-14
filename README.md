# Deep Learning Course Assignments – CSC2516 (University of Toronto)

This repository contains the completed assignments for **CSC2516: Neural Networks and Deep Learning**, a graduate-level course taught by **Professor Roger Grosse** at the University of Toronto.

## 📘 Course Description

Machine learning enables computers to learn patterns and make decisions from data without explicit programming. Deep learning, powered by neural networks, has driven major advances in image recognition, natural language processing, and reinforcement learning.

This course introduced both the theoretical foundations and practical tools of modern deep learning. It covered supervised learning methods (e.g., linear models, MLPs, CNNs, Transformers), followed by topics in unsupervised learning and reinforcement learning. Emphasis was placed on optimization, generalization, and implementation with PyTorch.

## 📝 Assignment Summaries

### Assignment 1 – Linear Models and Backpropagation
- Explored over/underparameterized linear regression and gradient descent convergence.
- Derived and implemented full backpropagation with Jacobian analysis.
- Computed gradient norms efficiently and constructed logic-inspired neural networks.

### Assignment 2 – Optimization and Image Colorization
- Analyzed optimization methods including SGD, RMSProp, and weight decay.
- Applied gradient-based hyperparameter tuning.
- Built CNNs to perform image colorization as a classification task on CIFAR-10.

### Assignment 3 – Transformers and NLP
- Trained Transformer models for neural machine translation (English → Pig Latin).
- Compared encoder-decoder and decoder-only architectures.
- Charted model performance scaling laws with IsoFLOP analysis.
- Fine-tuned BERT variants for verbal arithmetic classification.

## 📂 Repository Structure
```text
deep-learning-assignments/
├── assignment1/ # Linear models, backpropagation, gradient norms
├── assignment2/ # Optimization, CNNs, image colorization
├── assignment3/ # Transformers, NLP, fine-tuning
└── README.md # This file
```

## 💻 Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
