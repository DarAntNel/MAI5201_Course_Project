import os, time, fasttext, torch, numpy as np, pandas as pd
from transformers import BertTokenizer, BertModel
import kagglehub, shutil
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def get_bert_embeddings(texts, tokenizer, bert_model, device="cpu", batch_size=4):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = bert_model(**encoded_input)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(cls_embeddings.cpu().numpy())
            torch.cuda.empty_cache()
    return np.vstack(embeddings)

def embedding_to_tokens(embeddings, precision=2):
    token_texts = []
    for emb in embeddings:
        tokens = [f"dim{i}_{round(float(val), precision)}" for i, val in enumerate(emb)]
        token_texts.append(" ".join(tokens))
    return token_texts

def prepare_fasttext_file(texts, labels, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for text, label in zip(texts, labels):
            f.write(f"__label__{label} {text}\n")

def train_fasttext(train_file, model_path, **kwargs):
    if os.path.exists(model_path):
        print(f"[INFO] Loading existing FastText model from {model_path}")
        return fasttext.load_model(model_path), 0.0
    print(f"[INFO] Training FastText model for {train_file} ...")
    start_time = time.time()
    model = fasttext.train_supervised(input=train_file, **kwargs)
    duration = time.time() - start_time
    model.save_model(model_path)
    print(f"[INFO] Training completed in {duration:.2f}s for {train_file}")
    return model, duration

def evaluate_fasttext(model, texts, true_labels):
    preds = [model.predict(text)[0][0].replace("__label__", "") for text in texts]
    true_labels = [str(lbl) for lbl in true_labels]
    accuracy = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average="weighted")
    print("\n📊 Evaluation Metrics:")
    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1-score : {f1*100:.2f}%")
    return accuracy, precision, recall, f1

def record_results(dataset_name, model_type, train_time, acc, prec, rec, f1, csv_file="fasttext_results.csv"):
    df_new = pd.DataFrame([{
        "Dataset": dataset_name,
        "Model Type": model_type,
        "Training Time (s)": round(train_time, 2),
        "Accuracy": round(acc * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall": round(rec * 100, 2),
        "F1-score": round(f1 * 100, 2)
    }])
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(csv_file, index=False)
    print(f"[INFO] Results saved to {csv_file}")

def download_dataset(handle: str):
    cache_path = kagglehub.dataset_download(handle)
    project_data_dir = os.path.join("data", handle)
    os.makedirs(project_data_dir, exist_ok=True)
    shutil.copytree(cache_path, project_data_dir, dirs_exist_ok=True)
    print("Dataset copied to project folder:", project_data_dir)
    return project_data_dir


def load_and_normalize_dataset(csv_path):
    """
    Loads a CSV file that may or may not have headers and tries to detect
    which columns are text and which are labels.
    Returns: texts (list[str]), labels (list[str])
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, header=None)

    if df.columns[0] == 0 or df.columns[0] == "0":
        df.columns = [f"col_{i}" for i in range(df.shape[1])]

    possible_label_cols = [col for col in df.columns if any(k in str(col).lower() for k in ["class", "label", "category", "sentiment", "rating", "polarity"])]
    possible_text_cols = [col for col in df.columns if any(k in str(col).lower() for k in ["title", "text", "content", "description", "review", "body", "question"])]

    if not possible_text_cols or not possible_label_cols:
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            sample_val = str(df[col].iloc[0])
            if unique_ratio > 0.1 and len(sample_val.split()) > 3:
                possible_text_cols.append(col)
            elif unique_ratio < 0.1:
                possible_label_cols.append(col)

    if not possible_text_cols:
        raise ValueError(f"❌ Could not detect text column in {csv_path}. Columns: {list(df.columns)}")
    if not possible_label_cols:
        raise ValueError(f"❌ Could not detect label column in {csv_path}. Columns: {list(df.columns)}")

    text_col = possible_text_cols[0]
    label_col = possible_label_cols[0]

    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()

    return texts, labels


def process_dataset(handle, tokenizer, bert_model, device="cpu"):
    print(f"\n============================")
    print(f"📦 Processing dataset: {handle}")
    print(f"============================")


    if handle == "soumikrakshit/yahoo-answers-dataset":
        handle = "soumikrakshit/yahoo-answers-dataset/yahoo_answers_csv"
    elif handle == "irustandi/yelp-review-polarity":
        handle = "irustandi/yelp-review-polarity/yelp_review_polarity_csv"

    dataset_dir = os.path.join("data", handle)
    train_file_path = os.path.join(dataset_dir, "train.csv")
    test_file_path = os.path.join(dataset_dir, "test.csv")




    if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
        print(f"[WARNING] Missing train/test CSV in {dataset_dir}. Skipping...")
        return

    train_csv = os.path.join(dataset_dir, "train.csv")
    test_csv = os.path.join(dataset_dir, "test.csv")

    df_train, labels_train = load_and_normalize_dataset(train_csv)
    df_test, labels_test = load_and_normalize_dataset(test_csv)

    base_name = handle.replace("/", "_")
    train_emb_file = f"{base_name}_train_embeddings.npy"
    test_emb_file = f"{base_name}_test_embeddings.npy"
    ft_train_file = f"{base_name}_fasttext_train.txt"
    ft_test_file = f"{base_name}_fasttext_test.txt"
    ft_model_file = f"{base_name}_ft_model.bin"

    if os.path.exists(train_emb_file):
        embeddings = np.load(train_emb_file)
    else:
        embeddings = get_bert_embeddings(df_train, tokenizer, bert_model, device)
        np.save(train_emb_file, embeddings)

    if os.path.exists(test_emb_file):
        embeddings_test = np.load(test_emb_file)
    else:
        embeddings_test = get_bert_embeddings(df_test, tokenizer, bert_model, device)
        np.save(test_emb_file, embeddings_test)

    token_texts_train = embedding_to_tokens(embeddings)
    token_texts_test = embedding_to_tokens(embeddings_test)

    if not os.path.exists(ft_train_file):
        prepare_fasttext_file(token_texts_train, labels_train, ft_train_file)
    if not os.path.exists(ft_test_file):
        prepare_fasttext_file(token_texts_test, labels_test, ft_test_file)

    ft_model, train_time = train_fasttext(ft_train_file, model_path=ft_model_file, epoch=5, lr=1.0, wordNgrams=1, verbose=2)
    acc, prec, rec, f1 = evaluate_fasttext(ft_model, token_texts_test, labels_test)
    record_results(handle, "FastText (BERT-tokenized)", train_time, acc, prec, rec, f1)

    ft_train_raw = f"{base_name}_fasttext_train_raw.txt"
    ft_test_raw = f"{base_name}_fasttext_test_raw.txt"
    ft_model_raw = f"{base_name}_ft_model_raw.bin"

    if not os.path.exists(ft_train_raw):
        prepare_fasttext_file(df_train, labels_train, ft_train_raw)
    if not os.path.exists(ft_test_raw):
        prepare_fasttext_file(df_test, labels_test, ft_test_raw)

    ft_raw_model, train_time_raw = train_fasttext(ft_train_raw, model_path=ft_model_raw, epoch=5, lr=1.0, wordNgrams=2, verbose=2)
    acc_r, prec_r, rec_r, f1_r = evaluate_fasttext(ft_raw_model, df_test, labels_test)
    record_results(handle, "FastText (Raw text)", train_time_raw, acc_r, prec_r, rec_r, f1_r)



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()
    kaggle_datasets = [
        "amananandrai/ag-news-classification-dataset",
        "irustandi/yelp-review-polarity",
        # "jarvistian/sogou-news-corpus",
        # "yelp-dataset/yelp-dataset",
        "soumikrakshit/yahoo-answers-dataset",
        "kritanjalijain/amazon-reviews",
        "bhavikardeshna/amazon-customerreviews-polarity",
    ]

    for ds in kaggle_datasets:
        try:
            download_dataset(ds)
            process_dataset(ds, tokenizer, bert_model, device)
        except Exception as e:
            print(f"[ERROR] Failed on dataset {ds}: {e}")

