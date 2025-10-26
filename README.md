# MAI5201 Course Project: BERT vs. n-Grams for Text Classification

## Overview

This project investigates the performance of contextual embeddings (BERT, SBERT) versus traditional n-gram FastText embeddings for text classification.  
It uses Kaggle datasets to evaluate classification accuracy, precision, recall, and F1-score.

---

## Project Structure

MAI5201_Course_Project/
│
├─ main.py # Main script to run experiments
├─ requirements.txt # Python dependencies
├─ fasttext_results.csv # Stores experiment results
├─ data/ # Folder for downloaded datasets
└─ README.md



---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/DarAntNel/MAI5201_Course_Project.git
cd MAI5201_Course_Project

2. Create a Virtual Environment
python -m venv venv

3. Activate the Virtual Environment

Windows:

.\venv\Scripts\activate

macOS/Linux:

source venv/bin/activate

4. Install Dependencies
pip install -r requirements.txt

```

Running Experiments

Open main.py and locate the dataset list:

kaggle_datasets = [
    "amananandrai/ag-news-classification-dataset",
    "irustandi/yelp-review-polarity",
    "soumikrakshit/yahoo-answers-dataset",
    "kritanjalijain/amazon-reviews",
    "bhavikardeshna/amazon-customerreviews-polarity",
]


Important: Only enable one dataset at a time by commenting out the others.

Run the script:

python main.py

Below the main loop to run on these kaggle_datasets is code for single sentence testing you can uncomment it and comment the main loop to evaluate single testing (this can only be done after models have been created)