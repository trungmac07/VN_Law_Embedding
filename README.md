# Fine-Tuning Vietnamese Legal Document Retrieval

This repository contains the code and dataset used to fine-tune a language model for Embedding Similarity. The fine-tuned model is designed to enhance embedding similarity between questions related to Vietnamese law and relevant legal documents, enabling efficient retrieval for question-answering tasks.

## Table of Contents


- [Model Overview](#model-overview)
- [Dataset](#dataset)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Usage](#usage)




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

### Preprocessed Dataset

The preprocessed dataset is available on [Google Drive](https://drive.google.com/drive/folders/1LK6_fg9Q1m8D6auLGP_yYw_qVlho-bop?usp=sharing). It includes three main components: the corpus, questions, and the relationships between questions and documents.

- Files **with the number 2** represent data sourced from [`Legal Conversation v2`](https://huggingface.co/datasets/chillies/legal-conversation-v2).

- Files **without the number 2** represent data sourced from the [`Zalo AI 2021`](https://www.kaggle.com/datasets/hariwh0/zaloai2021-legal-text-retrieval/code) competition.

### Dataset Structure

- **`corpus` and `corpus_2`**: Contain document IDs and their corresponding document contents.
- **`questions` and `questions_2`**: Contain question IDs and their corresponding questions.
- **`qnc` and `qnc_2`**: Each row contains a question ID and a relevant document ID that can be used to answer the question.


- **Questions**: Legal queries in Vietnamese.
- **Documents**: Relevant legal documents or sections of legal documents.


## Training

To fine-tune model for embedding similarity between legal question and relevant documents, follow these steps:

1. Clone this repository to your local machine.
2. Upload your dataset to Google Drive.
3. Open the provided notebook in Google Colab to take advantage of robust GPU support and resources. Update the file paths in the notebook to point to your cloud-hosted dataset.
4. Execute the notebook to begin the training and evaluation process.



## Evaluation

The model's performance was evaluated using an Information Retrieval evaluator, focusing on its ability to accurately retrieve the correct legal document for a given question. Key evaluation metrics include:

- **Accuracy@K**: The proportion of queries for which the correct document is ranked within the top K results.
- **Recall@K**: The percentage of correct documents retrieved within the top K results.
- **Mean Reciprocal Rank (MRR)**: The average of the reciprocal ranks of the correct documents.


|Metrics|Base Model|Train on Zalo's dataset| Train on both datasets|
|-|-:|-:|-:|
| **dim_128_cosine_accuracy@1** | 0.390                | 0.532              | 0.701              |
| **dim_128_cosine_accuracy@3** | 0.600                 | 0.708              | 0.867              |
| **dim_128_cosine_accuracy@5** | 0.667               | 0.784              | 0.907              |
| **dim_128_cosine_accuracy@10**| 0.748               | 0.853              | 0.969              |
| **dim_128_cosine_precision@1**| 0.390                | 0.532              | 0.712              |
| **dim_128_cosine_precision@3**| 0.279               | 0.361              | 0.412              |
| **dim_128_cosine_precision@5**| 0.211               | 0.270              | 0.230              |
| **dim_128_cosine_precision@10**| 0.136              | 0.169              | 0.158              |
| **dim_128_cosine_recall@1**   | 0.223               | 0.302              | 0.470              |
| **dim_128_cosine_recall@3**   | 0.420              | 0.532              | 0.698              |
| **dim_128_cosine_recall@5**   | 0.508               | 0.633              | 0.772              |
| **dim_128_cosine_recall@10**  | 0.621               | 0.757              | 0.902             |
| **dim_128_cosine_ndcg@10**     | 0.496               | 0.628              | 0.793             |
| **dim_128_cosine_mrr@10**      | 0.511              | 0.636              | 0.810              |
| **dim_128_cosine_map@100**     | 0.434              | 0.565              | 0.727  