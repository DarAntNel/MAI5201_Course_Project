import os, time, fasttext, torch, numpy as np, pandas as pd
from transformers import BertTokenizer, BertModel
import kagglehub, shutil, zipfile
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import faiss


class HybridEmbeddingTokenizer:
    def __init__(self, n_clusters=4096, pca_dim=64, top_k=3, n_bins=10, gpu=False):
        self.n_clusters = n_clusters
        self.pca_dim = pca_dim
        self.top_k = top_k
        self.n_bins = n_bins
        self.gpu = gpu
        self.pca = PCA(n_components=self.pca_dim)
        self.centroids = None
        self.index = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        # PCA
        X_reduced = self.pca.fit_transform(X)

        # Faiss KMeans clustering
        kmeans = faiss.Kmeans(d=self.pca_dim, k=self.n_clusters, niter=50, verbose=True, gpu=self.gpu)
        kmeans.train(X_reduced)
        self.centroids = kmeans.centroids

        # FAISS index for nearest neighbor search
        self.index = faiss.IndexFlatL2(self.pca_dim)
        if self.gpu:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        self.index.add(self.centroids)

        # Setup PCA discretization bins
        self.kbins = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        self.kbins.fit(X_reduced)

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_reduced = self.pca.transform(X)

        # Soft clustering top-k
        distances, idxs = self.index.search(X_reduced, self.top_k)
        token_texts = []
        for i, (dist, ids) in enumerate(zip(distances, idxs)):
            weights = np.exp(-dist)
            weights /= weights.sum()
            cluster_tokens = [f"c{cid}:{w:.3f}" for cid, w in zip(ids, weights)]

            # PCA bin tokens
            bins = self.kbins.transform([X_reduced[i]])[0]
            bin_tokens = [f"pca{i}_bin{int(b)}" for i, b in enumerate(bins)]

            token_texts.append(" ".join(cluster_tokens + bin_tokens))
        return token_texts

def clean_embeddings(X):
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    non_zero_mask = np.any(X != 0.0, axis=1)
    if not np.all(non_zero_mask):
        print(f"[WARN] Removing {np.sum(~non_zero_mask)} empty or invalid embeddings")
    return X[non_zero_mask]


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


def embedding_to_tokens(embeddings, num_bins=10):
    tokens_list = []
    emb_min = embeddings.min(axis=0)
    emb_max = embeddings.max(axis=0)
    bin_width = (emb_max - emb_min + 1e-8) / num_bins

    for emb in embeddings:
        bins = ((emb - emb_min) / bin_width).astype(int)
        bins = np.clip(bins, 0, num_bins - 1)  # avoid out-of-range
        tokens = [f"dim{i}_bin{b}" for i, b in enumerate(bins)]
        tokens_list.append(" ".join(tokens))

    return tokens_list


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


def evaluate_fasttext(model, texts, true_labels, pretokenized=False):
    """
    Evaluates a FastText model.
    - texts: list of strings (raw or tokenized)
    - true_labels: list of strings
    - pretokenized: if True, texts are already tokenized (BERT/SBERT tokens)
    """
    preds = []

    for text in texts:
        # FastText expects a string; tokenized text is space-joined
        input_text = text if pretokenized else str(text)
        pred_label, _ = model.predict(input_text, k=1)
        pred_label = pred_label[0].replace("__label__", "")
        preds.append(pred_label)

    true_labels = [str(lbl) for lbl in true_labels]

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        preds,
        average="macro",
        zero_division=0
    )
    accuracy = accuracy_score(true_labels, preds)

    print("\n📊 Evaluation Metrics (Macro):")
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

    bert_embeddings = clean_embeddings(bert_embeddings)
    bert_embeddings_test = clean_embeddings(bert_embeddings_test)

    tokenizer = HybridEmbeddingTokenizer(
        n_clusters=4096,
        pca_dim=64,
        top_k=50,
        gpu=False
    )

    tokenizer.fit(bert_embeddings)

    bert_token_texts_train = tokenizer.transform(bert_embeddings)
    bert_token_texts_test = tokenizer.transform(bert_embeddings_test)

    if not os.path.exists(bert_ft_train_file):
        prepare_fasttext_file(bert_token_texts_train, labels_train, bert_ft_train_file)
    if not os.path.exists(bert_ft_test_file):
        prepare_fasttext_file(bert_token_texts_test, labels_test, bert_ft_test_file)

    bert_ft_model, bert_train_time = train_fasttext(bert_ft_train_file, model_path=bert_ft_model_file, epoch=5, lr=0.1,
                                                    wordNgrams=1, verbose=2)
    acc, prec, rec, f1 = evaluate_fasttext(bert_ft_model, bert_token_texts_test, labels_test, True)
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

    sbert_embeddings = clean_embeddings(sbert_embeddings)
    sbert_embeddings_test = clean_embeddings(sbert_embeddings_test)

    tokenizer = HybridEmbeddingTokenizer(
        n_clusters=4096,
        pca_dim=64,
        top_k=30,
        gpu=False
    )

    tokenizer.fit(sbert_embeddings)

    sbert_token_texts_train = tokenizer.transform(sbert_embeddings)
    sbert_token_texts_test = tokenizer.transform(sbert_embeddings_test)

    if not os.path.exists(sbert_ft_train_file):
        prepare_fasttext_file(sbert_token_texts_train, labels_train, sbert_ft_train_file)
    if not os.path.exists(sbert_ft_test_file):
        prepare_fasttext_file(sbert_token_texts_test, labels_test, sbert_ft_test_file)

    sbert_ft_model, sbert_train_time = train_fasttext(sbert_ft_train_file, model_path=sbert_ft_model_file, epoch=5,
                                                      lr=0.1, wordNgrams=1, verbose=2)
    acc_s, prec_s, rec_s, f1_s = evaluate_fasttext(sbert_ft_model, sbert_token_texts_test, labels_test, pretokenized=True)
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

    ft_raw_model, train_time_raw = train_fasttext(ft_train_raw, model_path=ft_model_raw, epoch=5, lr=0.1, wordNgrams=2,
                                                  verbose=2)
    acc_r, prec_r, rec_r, f1_r = evaluate_fasttext(ft_raw_model, df_test, labels_test, pretokenized=False)
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




    # "MANUAL TESTING DONE"
    # import fasttext
    #
    # # --- Specify your model path ---
    # model_path = "irustandi_yelp-review-polarity_yelp_review_polarity_csv_ft_model_raw.bin"  # change to the model you want
    # model = fasttext.load_model(model_path)
    #
    # # --- Test a single sentence ---
    # sentence = "Went here with a gift card from the restaurant week photo competition. We were one of two tables there and a bunch of staff showing up during the meal. Our waitress was good, but that couldn't save the food.The meal started out with some what I guess used to be foccacia.  I say used to be because it no longer represented anything other than long pieces of brick or bark very stale and tasteless.We ordered the vegetable board which was probably the best thing we ordered with light and bright vegetables perfectly cooked and paired with two delicious sauces. Then we ordered the octopus which was fine, but very cold and the octopus had little to no flavor, being overwhelmed by the citrus in the dish. The tuscan wings were ok, but they were really ordered for my husband who seemed to enjoy them. Finally the diver scallops.  This dish was indeible.  The scallops were very overcooked and the pea puree was so minty I was unable to eat it. We sent it back and they did take off half of the price (weren't expecting anything just didn't want to eat it). After that we decided against dessert and left.  I probably will not return."
    # # Predict label
    # pred_label, pred_prob = model.predict(sentence)
    # pred_label = pred_label[0].replace("__label__", "")
    # pred_prob = pred_prob[0]
    #
    # print(f"Input Sentence: {sentence}")
    # print(f"Predicted Label: {pred_label}")
    # print(f"Confidence: {pred_prob:.4f}")
    #
    # dataset_base = "irustandi_yelp-review-polarity_yelp_review_polarity_csv"
    #
    # # Paths
    # bert_emb_file = f"{dataset_base}_bert_train_embeddings.npy"
    # sbert_emb_file = f"{dataset_base}_sbert_train_embeddings.npy"
    #
    # bert_ft_model_file = f"{dataset_base}_bert_ft_model.bin"
    # sbert_ft_model_file = f"{dataset_base}_sbert_ft_model.bin"
    #
    # bert_tokenizer_file = f"{dataset_base}_bert_tokenizer.pkl"
    # sbert_tokenizer_file = f"{dataset_base}_sbert_tokenizer.pkl"
    #
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    #
    # # ------------------ LOAD FASTTEXT MODELS ------------------
    # bert_ft_model = fasttext.load_model(bert_ft_model_file)
    # sbert_ft_model = fasttext.load_model(sbert_ft_model_file)
    #
    # # ------------------ LOAD BERT/SBERT MODELS ------------------
    # bert_tokenizer_hf = BertTokenizer.from_pretrained("bert-base-uncased")
    # bert_model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()
    # sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    #
    # # ------------------ LOAD OR CREATE HYBRID TOKENIZERS ------------------
    # # BERT tokenizer
    # if os.path.exists(bert_tokenizer_file):
    #     with open(bert_tokenizer_file, "rb") as f:
    #         bert_hybrid_tokenizer = pickle.load(f)
    # else:
    #     print("[INFO] Creating BERT HybridEmbeddingTokenizer from training embeddings...")
    #     bert_embeddings = np.load(bert_emb_file)
    #     bert_embeddings = clean_embeddings(bert_embeddings)
    #     bert_hybrid_tokenizer = HybridEmbeddingTokenizer(n_clusters=4096, pca_dim=64, top_k=50, gpu=False)
    #     bert_hybrid_tokenizer.fit(bert_embeddings)
    #     with open(bert_tokenizer_file, "wb") as f:
    #         pickle.dump(bert_hybrid_tokenizer, f)
    #
    # # SBERT tokenizer
    # if os.path.exists(sbert_tokenizer_file):
    #     with open(sbert_tokenizer_file, "rb") as f:
    #         sbert_hybrid_tokenizer = pickle.load(f)
    # else:
    #     print("[INFO] Creating SBERT HybridEmbeddingTokenizer from training embeddings...")
    #     sbert_embeddings = np.load(sbert_emb_file)
    #     sbert_embeddings = clean_embeddings(sbert_embeddings)
    #     sbert_hybrid_tokenizer = HybridEmbeddingTokenizer(n_clusters=4096, pca_dim=64, top_k=30, gpu=False)
    #     sbert_hybrid_tokenizer.fit(sbert_embeddings)
    #     with open(sbert_tokenizer_file, "wb") as f:
    #         pickle.dump(sbert_hybrid_tokenizer, f)
    #
    # # ------------------ TEST SENTENCE ------------------
    #
    # # --- Helper: get BERT tokens ---
    # def sentence_to_bert_tokens(sentence):
    #     encoded_input = bert_tokenizer_hf([sentence], padding=True, truncation=True, return_tensors="pt").to(device)
    #     with torch.no_grad():
    #         outputs = bert_model(**encoded_input)
    #         cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    #     cls_embedding = np.nan_to_num(cls_embedding)
    #     token_text = bert_hybrid_tokenizer.transform(cls_embedding)[0]
    #     return token_text
    #
    #
    # # --- Helper: get SBERT tokens ---
    # def sentence_to_sbert_tokens(sentence):
    #     embedding = sbert_model.encode([sentence], convert_to_numpy=True)
    #     embedding = np.nan_to_num(embedding)
    #     token_text = sbert_hybrid_tokenizer.transform(embedding)[0]
    #     return token_text
    #
    #
    # # --- Get tokens ---
    # bert_tokens = sentence_to_bert_tokens(sentence)
    # sbert_tokens = sentence_to_sbert_tokens(sentence)
    #
    # # --- Predict labels ---
    # bert_pred_label = bert_ft_model.predict(bert_tokens)[0][0].replace("__label__", "")
    # sbert_pred_label = sbert_ft_model.predict(sbert_tokens)[0][0].replace("__label__", "")
    #
    # print(f"\nTest Sentence: {sentence}")
    # print(f"BERT FastText Prediction  : {bert_pred_label}")
    # print(f"SBERT FastText Prediction : {sbert_pred_label}")