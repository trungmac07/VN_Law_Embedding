# Fine-Tuning Vietnamese Embedding For Legal Documents Retrieval

<img src = "law.jpg" style = "display:block;
  margin-left: auto;
  margin-right: auto;
  width: 100%;}"
/>

This repository contains the code and dataset used to fine-tune a language model for Embedding Similarity. The fine-tuned model is designed to enhance embedding similarity between questions related to Vietnamese law and relevant legal documents, enabling efficient retrieval for question-answering tasks.

## Table of Contents


- [Overview](#overview)
- [Dataset](#dataset)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Usage](#usage)



## Overview

The base model used is [`hiieu/halong_embedding`](https://huggingface.co/hiieu/halong_embedding) which is fine-tuned from `intfloat/multilingual-e5-base` in Vietnamese dataset. 


The fine-tuning process leverages **Matryoshka Loss** combined with **Multi Negative Ranking Loss** to optimize the model for the task of document retrieval.

View the model [here](https://huggingface.co/truro7/vn-law-embedding)

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

The preprocessed dataset is available on [Kaggle](https://www.kaggle.com/datasets/trungmac/vn-law-embedding) or [Google Drive](https://drive.google.com/drive/folders/1LK6_fg9Q1m8D6auLGP_yYw_qVlho-bop?usp=sharing) . It includes three main components: the corpus, questions, and the relationships between questions and documents.

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

Due to the size of the model and dataset, I recommend using a platform such as Google Colab or Kaggle for robust GPU support.

1. Clone this repository to your local machine.
2. Upload your dataset to Kaggle (or Google Drive).
3. Open the provided notebook in Kaggle (or Google Colab) to take advantage of robust GPU support and resources. Update the file paths in the notebook to point to your dataset.
4. Execute the notebook to begin the training and evaluation process.

Visit my work on Kaggle [here](https://www.kaggle.com/code/trungmac/vn-law-embedding-fine-tuning)

Fine-tuned model [here](https://huggingface.co/truro7/vn-law-embedding)

## Evaluation

The model's performance was evaluated using an Information Retrieval evaluator, focusing on its ability to accurately retrieve the correct legal document for a given question. 


|Metrics|Base HaLong Embedding|HaLong Embedding on Zalo's dataset| VN_Law_Embedding on both dataset|
|-|-:|-:|-:|
| **dim_128_cosine_accuracy@1** | 0.390                | 0.532              | 0.623              |
| **dim_128_cosine_accuracy@3** | 0.600                 | 0.708              | 0.792              |
| **dim_128_cosine_accuracy@5** | 0.667               | 0.784              | 0.851              |
| **dim_128_cosine_accuracy@10**| 0.748               | 0.853              | 0.900              |
| **dim_128_cosine_precision@1**| 0.390                | 0.532              | 0.623              |
| **dim_128_cosine_precision@3**| 0.279               | 0.361              | 0.412              |
| **dim_128_cosine_precision@5**| 0.211               | 0.270              | 0.310              |
| **dim_128_cosine_precision@10**| 0.136              | 0.169              | 0.184              |
| **dim_128_cosine_recall@1**   | 0.223               | 0.302              | 0.353              |
| **dim_128_cosine_recall@3**   | 0.420              | 0.532              | 0.608              |
| **dim_128_cosine_recall@5**   | 0.508               | 0.633              | 0.722              |
| **dim_128_cosine_recall@10**  | 0.621               | 0.757              | 0.823             |
| **dim_128_cosine_ndcg@10**     | 0.496               | 0.628              | 0.706             |
| **dim_128_cosine_mrr@10**      | 0.511              | 0.636              | 0.717              |
| **dim_128_cosine_map@100**     | 0.434              | 0.565              | 0.645  


## Usage


### Direct usage

```python
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

model = SentenceTransformer("truro7/vn-law-embedding", truncate_dim = 128)

query = "Trộm cắp sẽ bị xử lý như thế nào?" 

corpus = """
[100_2015_QH13]
LUẬT HÌNH SỰ
Điều 173. Tội trộm cắp tài sản
Khoản 1:
1. Người nào trộm cắp tài sản của người khác trị giá từ 2.000.000 đồng đến dưới 50.000.000 đồng hoặc dưới 2.000.000 đồng nhưng thuộc một trong các trường hợp sau đây, thì bị phạt cải tạo không giam giữ đến 03 năm hoặc phạt tù từ 06 tháng đến 03 năm:
a) Đã bị xử phạt vi phạm hành chính về hành vi chiếm đoạt tài sản mà còn vi phạm;
b) Đã bị kết án về tội này hoặc về một trong các tội quy định tại các điều 168, 169, 170, 171, 172, 174, 175 và 290 của Bộ luật này, chưa được xóa án tích mà còn vi phạm;
c) Gây ảnh hưởng xấu đến an ninh, trật tự, an toàn xã hội;
d) Tài sản là phương tiện kiếm sống chính của người bị hại và gia đình họ; tài sản là kỷ vật, di vật, đồ thờ cúng có giá trị đặc biệt về mặt tinh thần đối với người bị hại.
    
"""
embedding = torch.tensor([model.encode(query)])
corpus_embeddings = torch.tensor([model.encode(corpus)])

cosine_similarities = F.cosine_similarity(embedding, corpus_embeddings)

print(cosine_similarities.item()) #0.81
```

### Retrieve top k documents
```python
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

model = SentenceTransformer("truro7/vn-law-embedding", truncate_dim = 128)
all_docs = read_all_docs() # Read all legal documents -> list of document contents 
top_k = 3
embedding_docs = torch.load(vectordb_path, weights_only=False).to(self.device) # Vector database

query = "Trộm cắp sẽ bị xử lý như thế nào?" 
embedding = torch.tensor(model.encode(query))
cosine_similarities = F.cosine_similarity(embedding.unsqueeze(0).expand(self.embedding_docs.shape[0], 1, 128), self.embedding_docs, dim = -1).view(-1)

top_k = cosine_similarities.topk(k)
top_k_indices = top_k.indices
top_k_values = top_k.values

print(top_k_values)  #Similarity scores
for i in top_k_indices:     #Show top k relevant documents
    print(all_docs[i])
    print("___________________________________________")
```