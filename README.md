📰 News Classification Transformer (PyTorch)

A complete Transformer-based text classification model built entirely from scratch using PyTorch for classifying news headlines into four categories from the AG News dataset.

Unlike implementations that rely on torch.nn.Transformer, this project manually implements the core Transformer components including:

Token Embeddings
Sinusoidal Positional Encoding
Multi-Head Self-Attention
Feed-Forward Networks
Residual Connections
Layer Normalization
Attention Masking

The project also includes training, evaluation, an interactive inference demo, and a classical machine learning baseline for comparison.

Features
Transformer implemented from scratch
Custom tokenizer and vocabulary builder
Manual sinusoidal positional encoding
Multi-Head Self-Attention implementation
Transformer encoder blocks
Padding masks for variable-length sequences
Mean pooling for sentence representation
Training and validation pipeline
Confusion matrix generation
Classification report
Interactive command-line demo
TF-IDF + Logistic Regression baseline comparison
Project Structure
.
├── attention_block.py     # Multi-head attention and Transformer block
├── baseline.py            # TF-IDF + Logistic Regression baseline
├── dataset.py             # AG News dataset loader
├── demo.py                # Interactive prediction demo
├── embedding.py           # Token embedding & positional encoding
├── evaluate.py            # Model evaluation
├── model.py               # Complete Transformer model
├── tokenizer.py           # Tokenizer and vocabulary builder
├── train.py               # Training script
├── best_model.pt          # Saved trained model
└── README.md
Dataset

This project uses the AG News dataset.

The dataset contains four classes:

World
Sports
Business
Sci/Tech

Each sample is formed by combining the article title and description into a single text sequence.

Model Architecture
Input Text
      │
      ▼
Tokenizer
      │
      ▼
Vocabulary Encoding
      │
      ▼
Token Embedding
      │
      ▼
Positional Encoding
      │
      ▼
Transformer Encoder
    ├── Multi-Head Self Attention
    ├── Feed Forward Network
    ├── Residual Connections
    └── Layer Normalization
      │
      ▼
Mean Pooling
      │
      ▼
Linear Classification Layer
      │
      ▼
Predicted News Category
Training Configuration
Parameter	Value
Embedding Dimension	64
Attention Heads	4
Feed Forward Dimension	256
Transformer Layers	2
Batch Size	64
Learning Rate	0.001
Epochs	5
Vocabulary Size	15,000
Maximum Sequence Length	60
Dropout	0.1
Training

Run:

python train.py

The script automatically:

Loads the AG News dataset
Builds the vocabulary
Creates DataLoaders
Trains the Transformer
Evaluates on the validation set
Saves the best model as:
best_model.pt
Evaluation

Run:

python evaluate.py

This generates:

Classification Report
Confusion Matrix
Overall validation performance

The confusion matrix is automatically saved as:

confusion_matrix.png
Interactive Demo

After training:

python demo.py

Example:

> Apple unveils its newest AI-powered MacBook

Prediction:
Sci/Tech (99.2%)
Baseline Comparison

For comparison, the project also includes a classical machine learning baseline using:

TF-IDF
Logistic Regression

Run:

python baseline.py

This provides a reference point to compare the custom Transformer against a traditional text classification pipeline.

Technologies Used
Python
PyTorch
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Learning Objectives

This project was built to gain a deeper understanding of the internal mechanics of Transformer models by implementing the architecture from scratch instead of relying on high-level libraries.

Key concepts explored include:

Self-Attention
Multi-Head Attention
Positional Encoding
Transformer Encoder Architecture
Text Tokenization
Vocabulary Construction
Sequence Padding
Attention Masks
Deep Learning for NLP
Future Improvements
Add learned positional embeddings
Support pretrained word embeddings
Mixed precision (FP16) training
Learning rate scheduler
Early stopping
Attention visualization
Model checkpointing
Support for larger Transformer architectures
Hyperparameter tuning
License

This project is intended for educational and research purposes.
