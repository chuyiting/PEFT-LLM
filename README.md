# Project Overview

This repository contains all the necessary files and scripts for fine-tuning language models using contrastive learning and related objectives. Below is a breakdown of the included files and their purposes.

## Directory Structure

### Dockerfile

- **Purpose**: Configures the environment for deployment on a remote cluster. Ensures a consistent and reproducible setup for all workflows.

### data/

- **Description**: Directory containing all the necessary datasets.
  - **Training data**: Used for model training.
  - **Test data**: Used for model evaluation.
  - **Cluster data**: Preprocessed cluster segmentation.

### Scripts and Notebooks

This project includes several Python scripts and Jupyter notebooks for data preparation, fine-tuning, validation, and analysis.

#### 1. `finetune_transformer.py`

- **Purpose**: Script for contrastive learning fine-tuning using the Transformers library.
- **Usage**: Optimizes the model to rank relevant outputs higher and distinguish closely related items.

#### 2. `finetune-preprocess.ipynb`

- **Purpose**: Notebook for preprocessing raw data.
- **Usage**: Cleans and prepares data for training.

#### 3. `statistics.ipynb`

- **Purpose**: Generates dataset statistics and insights.
- **Usage**: Provides visualizations and metrics to analyze data distributions.

#### 4. `validate_embedding_objective.py`

- **Purpose**: Validates the results of training by assessing alignment with the embedding objective.
- **Usage**: Ensures the embeddings perform as intended.

#### 5. `finetune_embedding_objective.py`

- **Purpose**: The main script for fine-tuning the embedding objective.
- **Usage**: Used as the final step in the pipeline to train embeddings effectively.

#### 6. `rerank.py`

- **Purpose**: Implements a reranking mechanism to improve output relevance.
- **Usage**: Adjusts the ranking of model predictions based on embedding scores.

#### 7. `finetune.py`

- **Purpose**: Fine-tunes the language model using a Language Model Objective.
