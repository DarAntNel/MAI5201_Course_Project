import os, time, fasttext, torch, numpy as np, pandas as pd
from transformers import BertTokenizer, BertModel
import kagglehub, shutil, zipfile
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sentence_transformers import SentenceTransformer


def get_sbert_embeddings(texts, sbert_model, device="cpu", batch_size=16):
    print(f"[INFO] Getting SBERT embeddings for {len(texts)} samples (batch={batch_size}) on {device}...")
    start_time = time.time()

    sbert_model.to(device)

    embeddings = sbert_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
        device=device
    )

    total_time = time.time() - start_time
    return embeddings, total_time


def get_bert_embeddings(texts, tokenizer, bert_model, device="cpu", batch_size=16):
    embeddings = []
    start_time = time.time()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = bert_model(**encoded_input)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(cls_embeddings.cpu().numpy())
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    return np.vstack(embeddings), total_time


def embedding_to_tokens(embeddings, precision=2):
    token_texts = []
    for emb in embeddings:
        tokens = [f"dim{i}_{round(float(val), precision)}" for i, val in enumerate(emb)]
        token_texts.append(" ".join(tokens))
    return token_texts


def prepare_fasttext_file(texts, labels, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for text, label in zip(texts, labels):
            text = str(text).strip()
            label = str(label).strip()
            if text:  # skip empty lines
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
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1-score : {f1 * 100:.2f}%")
    return accuracy, precision, recall, f1


def record_results(dataset_name, model_type, train_time, acc, prec, rec, f1, emb_time=0.0,
                   csv_file="fasttext_results.csv"):
    df_new = pd.DataFrame([{
        "Dataset": dataset_name,
        "Model Type": model_type,
        "Embedding Time (s)": round(emb_time, 2),
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

    # Handle zipped datasets
    if cache_path.endswith(".zip"):
        with zipfile.ZipFile(cache_path, 'r') as zip_ref:
            zip_ref.extractall(project_data_dir)
    else:
        shutil.copytree(cache_path, project_data_dir, dirs_exist_ok=True)

    print("Dataset copied to project folder:", project_data_dir)
    return project_data_dir


def load_and_normalize_dataset(csv_path):
    """
    Loads a CSV file and automatically:
    - Detects label column (low cardinality, e.g. class or sentiment)
    - Detects one or multiple text columns (title, body, question, answer, etc.)
    - Combines multiple text columns into one string if necessary
    Returns: texts (list[str]), labels (list[str])
    """
    import pandas as pd

    # --- 1. Load CSV gracefully ---
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, header=None)

    # --- 2. Generate column names if numeric ---
    if df.columns[0] == 0 or df.columns[0] == "0":
        df.columns = [f"col_{i}" for i in range(df.shape[1])]

    # --- 3. Heuristic: detect text & label columns ---
    text_keywords = ["title", "text", "content", "description", "review", "body", "question", "answer", "passage"]
    label_keywords = ["class", "label", "category", "sentiment", "rating", "polarity"]

    possible_label_cols = [c for c in df.columns if any(k in str(c).lower() for k in label_keywords)]
    possible_text_cols = [c for c in df.columns if any(k in str(c).lower() for k in text_keywords)]

    # --- 4. Fallback detection based on content ---
    if not possible_text_cols or not possible_label_cols:
        for col in df.columns:
            unique_ratio = df[col].nunique() / max(len(df), 1)
            avg_len = df[col].astype(str).apply(lambda x: len(x.split())).mean()

            if unique_ratio > 0.1 and avg_len > 3:
                if col not in possible_text_cols:
                    possible_text_cols.append(col)
            elif unique_ratio < 0.3:
                if col not in possible_label_cols:
                    possible_label_cols.append(col)

    if not possible_text_cols:
        raise ValueError(f"❌ Could not detect text columns in {csv_path}. Columns: {list(df.columns)}")
    if not possible_label_cols:
        raise ValueError(f"❌ Could not detect label columns in {csv_path}. Columns: {list(df.columns)}")

    # --- 5. Combine multiple text columns ---
    df["__combined_text__"] = df[possible_text_cols].astype(str).agg(" ".join, axis=1)

    # --- 6. Choose label column (first one is usually correct) ---
    label_col = possible_label_cols[0]

    texts = df["__combined_text__"].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()

    return texts, labels


def process_dataset(handle, tokenizer, bert_model, sbert_model, device="cpu"):
    print(f"\n============================")
    print(f"📦 Processing dataset: {handle} on {device}")
    print(f"============================")

    # Adjust handle paths for nested datasets
    if handle == "soumikrakshit/yahoo-answers-dataset":
        handle = "soumikrakshit/yahoo-answers-dataset/yahoo_answers_csv"
    elif handle == "irustandi/yelp-review-polarity":
        handle = "irustandi/yelp-review-polarity/yelp_review_polarity_csv"

    dataset_dir = os.path.join("data", handle)
    train_file_path = os.path.join(dataset_dir, "train.csv")
    test_file_path = os.path.join(dataset_dir, "test.csv")

    if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Missing train.csv or test.csv in {dataset_dir}. Please check dataset structure.")

    df_train, labels_train = load_and_normalize_dataset(train_file_path)
    df_test, labels_test = load_and_normalize_dataset(test_file_path)

    base_name = handle.replace("/", "_")

    print(f"---------- BERT Embeddings ----------")
    bert_train_emb_file = f"{base_name}_bert_train_embeddings.npy"
    bert_test_emb_file = f"{base_name}_bert_test_embeddings.npy"
    bert_ft_train_file = f"{base_name}_bert_fasttext_train.txt"
    bert_ft_test_file = f"{base_name}_bert_fasttext_test.txt"
    bert_ft_model_file = f"{base_name}_bert_ft_model.bin"

    if os.path.exists(bert_train_emb_file):
        bert_embeddings = np.load(bert_train_emb_file)
        emb_train_time = 0.0
    else:
        bert_embeddings, emb_train_time = get_bert_embeddings(df_train, tokenizer, bert_model, device)
        np.save(bert_train_emb_file, bert_embeddings)

    if os.path.exists(bert_test_emb_file):
        bert_embeddings_test = np.load(bert_test_emb_file)
        emb_test_time = 0.0
    else:
        bert_embeddings_test, emb_test_time = get_bert_embeddings(df_test, tokenizer, bert_model, device)
        np.save(bert_test_emb_file, bert_embeddings_test)

    bert_total_emb_time = emb_train_time

    bert_token_texts_train = embedding_to_tokens(bert_embeddings)
    bert_token_texts_test = embedding_to_tokens(bert_embeddings_test)

    if not os.path.exists(bert_ft_train_file):
        prepare_fasttext_file(bert_token_texts_train, labels_train, bert_ft_train_file)
    if not os.path.exists(bert_ft_test_file):
        prepare_fasttext_file(bert_token_texts_test, labels_test, bert_ft_test_file)

    bert_ft_model, bert_train_time = train_fasttext(bert_ft_train_file, model_path=bert_ft_model_file, epoch=5, lr=1.0,
                                                    wordNgrams=1, verbose=2)
    acc, prec, rec, f1 = evaluate_fasttext(bert_ft_model, bert_token_texts_test, labels_test)
    record_results(handle, "FastText (BERT-tokenized)", bert_train_time, acc, prec, rec, f1,
                   emb_time=bert_total_emb_time)

    print(f"---------- SBERT Embeddings ----------")
    sbert_train_emb_file = f"{base_name}_sbert_train_embeddings.npy"
    sbert_test_emb_file = f"{base_name}_sbert_test_embeddings.npy"
    sbert_ft_train_file = f"{base_name}_sbert_fasttext_train.txt"
    sbert_ft_test_file = f"{base_name}_sbert_fasttext_test.txt"
    sbert_ft_model_file = f"{base_name}_sbert_ft_model.bin"

    if os.path.exists(sbert_train_emb_file):
        sbert_embeddings = np.load(sbert_train_emb_file)
        sbert_emb_train_time = 0.0
    else:
        sbert_embeddings, sbert_emb_train_time = get_sbert_embeddings(df_train, sbert_model, device)
        np.save(sbert_train_emb_file, sbert_embeddings)

    if os.path.exists(sbert_test_emb_file):
        sbert_embeddings_test = np.load(sbert_test_emb_file)
        sbert_emb_test_time = 0.0
    else:
        sbert_embeddings_test, sbert_emb_test_time = get_sbert_embeddings(df_test, sbert_model, device)
        np.save(sbert_test_emb_file, sbert_embeddings_test)

    sbert_total_emb_time = sbert_emb_train_time

    sbert_token_texts_train = embedding_to_tokens(sbert_embeddings)
    sbert_token_texts_test = embedding_to_tokens(sbert_embeddings_test)

    if not os.path.exists(sbert_ft_train_file):
        prepare_fasttext_file(sbert_token_texts_train, labels_train, sbert_ft_train_file)
    if not os.path.exists(sbert_ft_test_file):
        prepare_fasttext_file(sbert_token_texts_test, labels_test, sbert_ft_test_file)

    sbert_ft_model, sbert_train_time = train_fasttext(sbert_ft_train_file, model_path=sbert_ft_model_file, epoch=5,
                                                      lr=1.0, wordNgrams=1, verbose=2)
    acc_s, prec_s, rec_s, f1_s = evaluate_fasttext(sbert_ft_model, sbert_token_texts_test, labels_test)
    record_results(handle, "FastText (SBERT-tokenized)", sbert_train_time, acc_s, prec_s, rec_s, f1_s,
                   emb_time=sbert_total_emb_time)

    print(f"---------- FASTTEXT NGRAM ----------")
    ft_train_raw = f"{base_name}_fasttext_train_raw.txt"
    ft_test_raw = f"{base_name}_fasttext_test_raw.txt"
    ft_model_raw = f"{base_name}_ft_model_raw.bin"

    if not os.path.exists(ft_train_raw):
        prepare_fasttext_file(df_train, labels_train, ft_train_raw)
    if not os.path.exists(ft_test_raw):
        prepare_fasttext_file(df_test, labels_test, ft_test_raw)

    ft_raw_model, train_time_raw = train_fasttext(ft_train_raw, model_path=ft_model_raw, epoch=5, lr=1.0, wordNgrams=2,
                                                  verbose=2)
    acc_r, prec_r, rec_r, f1_r = evaluate_fasttext(ft_raw_model, df_test, labels_test)
    record_results(handle, "FastText (Raw text)", train_time_raw, acc_r, prec_r, rec_r, f1_r, emb_time=0.0)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

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
            process_dataset(ds, tokenizer, bert_model, sbert_model, device)
        except Exception as e:
            print(f"[ERROR] Failed on dataset {ds}: {e}")
