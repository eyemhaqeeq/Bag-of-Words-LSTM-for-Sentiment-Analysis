Bag of Words + LSTM for Sentiment Analysis

This project performs sentiment analysis on a dataset of Amazon product reviews using a Long Short-Term Memory (LSTM) neural network. Instead of using pre-trained embeddings like Word2Vec, it employs a Bag of Words (BoW) representation for text input. The goal is to empirically evaluate the effectiveness of the BoW approach when used with LSTM, and compare it against other input encoding strategies in an empirical study.

Objective
The core objective of this experiment is to analyze how well the LSTM model performs sentiment classification when trained on Bag of Words features. This experiment is part of a broader dissertation project focused on the empirical evaluation of input representation methods in sentiment analysis using deep learning.

 Methodology
1. Data Preprocessing
Dataset: Amazon product reviews

Preprocessing steps:

Lowercasing

Removing stopwords and punctuation

Tokenization

Lemmatization

2. Feature Representation
Text data was vectorized using CountVectorizer from scikit-learn

A fixed vocabulary size of 5000 most frequent words was used

The output matrix was reshaped into a 3D format suitable for LSTM input: (samples, 1, features)

3. Model Architecture
Framework: Keras (TensorFlow backend)

 Architecture:

One LSTM layer with 64 units

One Dense output layer with sigmoid activation

- Compilation:

Loss: Binary Crossentropy

Optimizer: Adam

Metrics: Accuracy

4. Training Details
Epochs: 5

Batch size: 64

Train/test split: 80/20

Evaluation Results(for 15000 samples)

Accuracy: 0.9056666666666666
Training Time: 8.126343965530396
Testing Time: 0.7066547870635986

- Additional Metrics:
Precision, Recall, F1-score (via classification_report)

Confusion Matrix Visualization

Training/Validation Accuracy and Loss plots

- Observations
The Bag of Words approach, although simplistic, achieved comparable accuracy to a Word2Vec-based LSTM in this setup.

This suggests that for certain datasets (especially when sequences are not exploited), BoW can be a strong baseline.

This also emphasizes the importance of evaluating not just the architecture (like LSTM) but the representation of input text data.

- Conclusion
The Bag of Words + LSTM model demonstrated competitive performance for binary sentiment classification. This experiment supports the broader hypothesis that input representation has a significant impact on the effectiveness of deep learning models, and should be carefully chosen based on the task and data characteristics.

This implementation can serve as a foundational baseline for comparing more advanced representations such as:

Word2Vec (mean-pooled and sequence-based)

GloVe

Contextual embeddings (e.g., BERT)
