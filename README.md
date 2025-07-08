# Custom Neural Language Models and MLP Experiments

This repository contains experiments and implementations for three core components:

1. **Next-Word Prediction with Custom Neural Language Models**
2. **Decision Boundary Visualizations with MLPs and Regularization**
3. **Digit Classification using MLPs on MNIST and Fashion-MNIST**

This work was completed as part of the course **ES 335: Machine Learning** at **IIT Gandhinagar**, taught by **Prof. Nipun Batra**, during Oct–Nov 2024.

Contributors: [Aryan Solanki](https://github.com/Aryan-IIT), [Nupoor Assudani](https://github.com/nupoor-ka), [Pranav Thakkar](https://github.com/Pranav4860), [Rishabh Jogani](https://github.com/rishabhh-7)

---

## 1. Next-Word Prediction with Neural Language Models

We built a next-word prediction model from scratch in PyTorch. Key components:

- **Vocabulary of 18,355 words** created after preprocessing and tokenizing input text.
- Trained **16 model variants** by varying:
  - Token (context) length - 8, 16
  - Embedding size - 16, 32
  - Activation functions - sin, ReLU
  - Random seed - 42, 96
- Visualized **word embeddings** using **t-SNE** to interpret semantic clustering.
- Deployed an **interactive Streamlit app** to allow users to test predictions and compare model behaviors.

---

## 2. MLP Decision Boundaries with Regularization

We created synthetic datasets and trained MLP classifiers to visualize decision boundaries under various settings:

- **No regularization (baseline MLP)**
- **L1 regularization**
- **L2 regularization**
- **Logistic regression** with polynomial feature expansion

We observed and compared the generalization and overfitting behavior across all models.

---

## 3. MLPs on MNIST and Fashion-MNIST

We trained an **MLP with architecture (30 → 20 → 10)** on the **MNIST dataset** (60,000 train, 10,000 test images).

- Compared with **Random Forest** and **Logistic Regression** classifiers.
- Evaluated performance using **F1-score** and **confusion matrices**.
- Analyzed **which digits are commonly confused**.

### t-SNE Embedding Visualizations

We performed **t-SNE** on the activations from the 20-neuron hidden layer:

- For a **trained MLP** vs. an **untrained MLP** → Trained embeddings form distinct clusters per digit.
- On the **Fashion-MNIST dataset** → Less separation, highlighting dataset mismatch and generalization limits.
