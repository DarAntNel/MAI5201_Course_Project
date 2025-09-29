from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import kagglehub, shutil, os
import numpy as np
import pandas as pd
import fasttext
import torch


def get_bert_embeddings(texts, batch_size=32):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = bert_model(**encoded_input)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(cls_embeddings.cpu().numpy())
    return np.vstack(embeddings)

def embedding_to_tokens(embeddings, precision=2):
    # Round embeddings and convert each dimension into a string token
    token_texts = []
    for emb in embeddings:
        tokens = [f"dim{i}_{round(float(val), precision)}" for i, val in enumerate(emb)]
        token_texts.append(" ".join(tokens))
    return token_texts


def prepare_fasttext_file(texts, labels, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for text, label in zip(texts, labels):
            f.write(f"__label__{label} {text}\n")


def download_dataset(handle: str):
    cache_path = kagglehub.dataset_download(handle)
    project_data_dir = os.path.join("data", handle)
    os.makedirs(project_data_dir, exist_ok=True)
    shutil.copytree(cache_path, project_data_dir, dirs_exist_ok=True)
    print("Dataset copied to project folder:", project_data_dir)

kaggle_datasets = [
    "amananandrai/ag-news-classification-dataset",
    "irustandi/yelp-review-polarity",
    "jarvistian/sogou-news-corpus",
    "yelp-dataset/yelp-dataset",
    "soumikrakshit/yahoo-answers-dataset",
    "kritanjalijain/amazon-reviews",
    "bhavikardeshna/amazon-customerreviews-polarity",
]

for ds in kaggle_datasets:
    download_dataset(ds)

# Handle Hugging Face dataset separately (saved as CSV)
hf_dataset = load_dataset("fancyzhx/dbpedia_14")
project_data_dir = "data/fancyzhx/dbpedia_14"
os.makedirs(project_data_dir, exist_ok=True)

for split, dset in hf_dataset.items():
    csv_path = os.path.join(project_data_dir, f"{split}.csv")
    dset.to_csv(csv_path)
    print(f"Saved {split} split to {csv_path}")



df = pd.read_csv("data/amananandrai/ag-news-classification-dataset/train.csv")
texts = (df['Title'] + " " + df['Description']).tolist()
labels = df['Class Index'].tolist()

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()


print("__get_bert_embeddings__")
embeddings = get_bert_embeddings(texts)
print("__Finished_get_bert_embeddings__")

print("__token_texts__")
token_texts = embedding_to_tokens(embeddings)
print("__Finished_token_texts__")

print("__prepare_fasttext_file__")
prepare_fasttext_file(token_texts, labels, "fasttext_train_bert.txt")
print("__Finished_prepare_fasttext_file__")

print("__Training__")
ft_model = fasttext.train_supervised(
    input="fasttext_train_bert.txt",
    epoch=5,
    lr=1.0,
    wordNgrams=1,
    verbose=2
)

print("__Finished__Training__")





df_test = pd.read_csv("data/amananandrai/ag-news-classification-dataset/test.csv")
texts_test = (df_test['Title'] + " " + df_test['Description']).tolist()
labels_test = df_test['Class Index'].tolist()

embeddings_test = get_bert_embeddings(texts_test)

token_texts_test = embedding_to_tokens(embeddings_test)

def prepare_fasttext_test_file(token_texts, labels, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for text, label in zip(token_texts, labels):
            f.write(f"__label__{label} {text}\n")

prepare_fasttext_test_file(token_texts_test, labels_test, "fasttext_test.txt")


ft_results = ft_model.test("fasttext_test.txt")
print(f"Number of samples: {ft_results[0]}")
print(f"Precision @1: {ft_results[1]:.4f}")
print(f"Recall @1: {ft_results[2]:.4f}")


