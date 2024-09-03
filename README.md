# Fine-Tuning Vietnamese Legal Document Retrieval

This repository contains the code and dataset used to fine-tune a language model for Embedding Similarity. The fine-tuned model is designed to enhance embedding similarity between questions related to Vietnamese law and relevant legal documents, enabling efficient retrieval for question-answering tasks.

## Table of Contents


- [Model Overview](#model-overview)
- [Dataset](#dataset)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)
- [License](#license)



## Overview

The base model used is [`hiieu/halong_embedding`](https://huggingface.co/hiieu/halong_embedding) which is fine-tuned from `intfloat/multilingual-e5-base` in Vietnamese dataset. 



The fine-tuning process leverages **Matryoshka Loss** combined with **Multi Negative Ranking Loss** to optimize the model for the task of document retrieval.

### Loss Functions

- **Matryoshka Loss**: Ensures that the embeddings of related entities are hierarchically structured, promoting better similarity scores for related queries and documents.
- **Multi Negative Ranking Loss**: Introduces multiple negative samples during training to improve the robustness of the model in distinguishing between relevant and irrelevant documents.

## Dataset

The raw dataset take from:
- [`Legal conversation v2`](https://huggingface.co/datasets/chillies/legal-conversation-v2): Contains pairs of a question and a corresponding answer by a human lawyer.
- [`Zalo AI 2021`](https://www.kaggle.com/datasets/hariwh0/zaloai2021-legal-text-retrieval/code): Contains queries, corpus and relationship between queries with its relevant documents.

The dataset used for fine-tuning was synthetically generated and preprocessed by myself. It is specifically tailored for the task of question-document similarity within the context of Vietnamese law.

This dataset consists of carefully constructed pairs of legal questions and their corresponding documents, ensuring it is highly relevant for training the model on this specific task.

### Dataset Structure

- **Questions**: Legal queries in Vietnamese.
- **Documents**: Relevant legal documents or sections of legal documents.

## Evaluation


The model's performance was evaluated using an Information Retrieval evaluator, focusing on its ability to accurately retrieve the correct legal document for a given question. Key evaluation metrics include:

- **Accuracy@K**: The proportion of queries for which the correct document is ranked within the top K results.
- **Recall@K**: The percentage of correct documents retrieved within the top K results.
- **Mean Reciprocal Rank (MRR)**: The average of the reciprocal ranks of the correct documents.

## Usage

To use the fine-tuned model for retrieving legal documents based on a question, follow these steps:

1. Clone this repository to your local machine.
2. Upload your dataset to a cloud storage service, such as Google Drive.
3. Open the provided notebook in Google Colab to take advantage of robust GPU support and resources. Update the file paths in the notebook to point to your cloud-hosted dataset.
4. Execute the notebook to begin the training and evaluation process.


