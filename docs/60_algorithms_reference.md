# 60 Machine Learning Algorithms — Project Reference

This document records the intended scope of the source guidebook and clarifies which entries are executable in the current tabular prototype.

## A. 40 classical scikit-learn algorithms

| ID | Algorithm | Task | Available in prototype |
|---:|---|---|---|
| 01 | Linear Regression | Regression | Yes |
| 02 | Ridge Regression | Regression | Yes |
| 03 | Lasso Regression | Regression | Yes |
| 04 | Elastic Net | Regression | Yes |
| 05 | Logistic Regression | Classification | Yes |
| 06 | SGD Classifier | Classification | Yes |
| 07 | Perceptron | Classification | Yes |
| 08 | Passive Aggressive Classifier | Classification | Yes |
| 09 | K-Nearest Neighbors Classifier | Classification | Yes |
| 10 | K-Nearest Neighbors Regressor | Regression | Yes |
| 11 | Support Vector Classifier | Classification | Yes |
| 12 | Support Vector Regression | Regression | Yes |
| 13 | Decision Tree Classifier | Classification | Yes |
| 14 | Decision Tree Regressor | Regression | Yes |
| 15 | Random Forest Classifier | Classification | Yes |
| 16 | Random Forest Regressor | Regression | Yes |
| 17 | Extra Trees Classifier | Classification | Yes |
| 18 | Extra Trees Regressor | Regression | Yes |
| 19 | Gradient Boosting Classifier | Classification | Yes |
| 20 | Gradient Boosting Regressor | Regression | Yes |
| 21 | Histogram Gradient Boosting Classifier | Classification | Yes |
| 22 | Histogram Gradient Boosting Regressor | Regression | Yes |
| 23 | AdaBoost Classifier | Classification | Yes |
| 24 | Bagging Classifier | Classification | Yes |
| 25 | Voting Classifier | Classification | Yes |
| 26 | Stacking Classifier | Classification | Yes |
| 27 | Gaussian Naive Bayes | Classification | Yes |
| 28 | Multinomial Naive Bayes | Classification | Yes |
| 29 | Bernoulli Naive Bayes | Classification | Yes |
| 30 | Linear Discriminant Analysis | Classification | Yes |
| 31 | Quadratic Discriminant Analysis | Classification | Yes |
| 32 | K-Means | Clustering | Yes |
| 33 | Mini-Batch K-Means | Clustering | Yes |
| 34 | DBSCAN | Clustering | Yes |
| 35 | Agglomerative Clustering | Clustering | Yes |
| 36 | Spectral Clustering | Clustering | Yes |
| 37 | Gaussian Mixture Model | Clustering | Yes |
| 38 | Principal Component Analysis | Dimensionality reduction | Yes |
| 39 | Truncated SVD | Dimensionality reduction | Yes |
| 40 | Isolation Forest | Anomaly detection | Yes |

## B. 10 PyTorch architecture/examples

| ID | Architecture/example | Typical input modality | Tabular prototype |
|---:|---|---|---|
| 41 | MLP classifier | Tabular numerical features | Not registered yet |
| 42 | MLP regressor | Tabular numerical features | Not registered yet |
| 43 | Small CNN | Image tensors | No |
| 44 | Simple RNN | Sequence tensors | No |
| 45 | LSTM | Sequence tensors | No |
| 46 | GRU | Sequence tensors | No |
| 47 | Autoencoder | Numerical vectors / feature representation | Not registered yet |
| 48 | Transformer encoder | Sequence tensors | No |
| 49 | Embedding + LSTM | Token sequences | No |
| 50 | GAN (generator + discriminator) | Distribution / generative samples | No |

## C. 10 TensorFlow/Keras architecture/examples

| ID | Architecture/example | Typical input modality | Tabular prototype |
|---:|---|---|---|
| 51 | Sequential MLP classifier | Tabular numerical features | Not registered yet |
| 52 | Functional API MLP | Tabular / multi-input | Not registered yet |
| 53 | MLP regressor | Tabular numerical features | Not registered yet |
| 54 | CNN image classifier | Image tensors | No |
| 55 | SimpleRNN classifier | Sequence tensors | No |
| 56 | LSTM classifier | Sequence tensors | No |
| 57 | GRU classifier | Sequence tensors | No |
| 58 | Bidirectional LSTM | Sequence tensors | No |
| 59 | Autoencoder | Numerical vectors / feature representation | Not registered yet |
| 60 | Transformer encoder | Sequence tensors | No |

## Important interpretation

The 60 entries are not 60 independent mathematical algorithms. Some entries are the same broad architecture implemented in different frameworks, while others solve different problem families or operate on different input modalities.

The working prototype activates the 40 scikit-learn classical entries because the initial product scope is CSV/XLSX tabular data. MLP and autoencoder variants may be added to tabular execution in a later deep-learning phase. CNN, RNN, LSTM, GRU, Transformer, token embedding, and GAN examples need specific image/sequence/token/generative data contracts and should not be forced into a generic flat-table workflow.
