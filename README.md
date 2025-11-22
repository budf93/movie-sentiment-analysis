# 🎬 GNN Sentiment Analysis

A Graph Neural Network (GNN) project for sentiment analysis on IMDB movie reviews. Unlike traditional NLP models that treat text as a sequence, this project treats text as a **graph structure**, connecting words based on proximity to capture contextual relationships.

The project includes a complete pipeline: **Data Preprocessing → GNN Training → Interactive Web Deployment**.

---

## 📌 Project Overview

* **Goal:** Classify movie reviews as **Positive** or **Negative**.
* **Method:** Converts text into graph structures where:
    * **Nodes:** Words in the review.
    * **Edges:** Connections between words within a sliding window context.
* **Model:** A Graph Convolutional Network (GCN) built with PyTorch Geometric.
* **Interface:** A Streamlit web app to test the model on custom text and visualize the underlying graph.

---

## 📦 Dependencies

The project relies on the following core libraries defined in `requirements.txt`:

* **`torch`**: The PyTorch deep learning framework used to build and train the GNN.
* **`torch_geometric`**: (Required) The extension library for PyTorch specifically for handling graph data.
* **`pandas`**: For loading and manipulating the IMDB dataset CSV.
* **`matplotlib`**: For visualizing the graph structure in the web app.
* **`streamlit`**: For creating the interactive web interface.

> **Note:** You will also need `networkx` for the graph visualization features.

---

## 🚀 Installation

1.  **Clone the repository** (or download the files):
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a virtual environment** (Recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install torch_geometric networkx
    ```

4.  **Download the Dataset**:
    * Download the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
    * Place the `IMDB Dataset.csv` file in the root project folder.

---

## 🛠️ Usage

### 1. Train the Model
Before running the app, you must train the model and generate the vocabulary.

Run the training script:
```bash
python train.py
````

  * **What this does:**
      * Loads the IMDB CSV data.
      * Builds a vocabulary from the text.
      * Converts reviews into graph data objects.
      * Trains the GNN model.
      * Saves `gnn_sentiment_model.pth` (model weights) and `vocab.pkl` (vocabulary) to your folder.

### 2\. Run the Web App

Once training is complete and the `.pth` and `.pkl` files exist, launch the Streamlit app:

```bash
streamlit run app.py
```

  * **What this does:**
      * Opens a local web server (usually at `http://localhost:8501`).
      * Allows you to type in any movie review.
      * Displays the **predicted sentiment** (Positive/Negative).
      * **Visualizes the graph** constructed from your input text.

-----

## 📂 File Structure

```
├── IMDB Dataset.csv        # (Download this from Kaggle)
├── train.py                # Script to process data and train the GNN
├── app.py                  # Streamlit application for inference
├── requirements.txt        # List of python dependencies
├── gnn_sentiment_model.pth # (Generated) Saved model weights
├── vocab.pkl               # (Generated) Saved vocabulary mapping
└── README.md               # Project documentation
```



### Why this format works
* **Clear Separation:** It separates "Installation" (setup) from "Usage" (running the code), which prevents confusion.
* **Explains the "Why":** It details exactly *why* `torch`, `pandas`, and `streamlit` are in the `requirements.txt`.
* **Step-by-Step:** It explicitly mentions running `train.py` *before* `app.py`, ensuring users don't get a "File Not Found" error.
