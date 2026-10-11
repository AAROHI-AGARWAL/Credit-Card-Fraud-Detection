# Credit Card Fraud Detection Using Anomaly Detection Techniques

## About the Project

This project detects potentially suspicious credit card transactions using anomaly detection techniques. It compares **Isolation Forest** and a **PyTorch Autoencoder** to identify unusual transaction patterns in a highly imbalanced dataset.

The project includes a machine learning pipeline, a FastAPI prediction API, an interactive Streamlit dashboard, and an SQLite database for storing transaction analysis history.

## Key Features

- Data preprocessing and exploratory data analysis (EDA)
- Isolation Forest anomaly detection
- Autoencoder-based anomaly detection using PyTorch
- Model evaluation using Precision, Recall, F1-score, and PR-AUC
- FastAPI endpoint for transaction predictions
- Streamlit dashboard for interactive analysis
- SQLite database for transaction history
- Risk distribution chart and summary metrics

## Technologies Used

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Deep Learning:** PyTorch
- **API:** FastAPI, Uvicorn
- **Dashboard:** Streamlit
- **Database:** SQLite
- **Visualization:** Matplotlib, Seaborn

## Dataset

The project uses the Credit Card Fraud Detection dataset containing anonymized transaction features.

- Original dataset: 284,807 transactions
- Original fraud cases: 492
- Features include `Time`, `V1`–`V28`, and `Amount`
- Target column: `Class` (0 = legitimate, 1 = fraud)

The dataset is highly imbalanced, so PR-AUC is used alongside other evaluation metrics.

Dataset source: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Model Performance

The following results were obtained on the held-out test set using thresholds selected on the validation set.

| Metric | Isolation Forest | Autoencoder |
|---|---:|---:|
| PR-AUC | 0.0712 | 0.2155 |
| Precision | 0.1346 | 0.3059 |
| Recall | 0.2211 | 0.2737 |
| F1-score | 0.1673 | 0.2889 |
| False Positives | 135 | 59 |
| False Negatives | 74 | 69 |

On this test set, the Autoencoder achieved better results across the reported metrics. These results do not guarantee the same performance on other data.

## Project Structure

```text
Credit-Card-Fraud-Detection/
├── api/
│   └── main.py
├── dashboard/
│   └── app.py
├── database/
│   ├── database.py
│   └── view_database.py
├── data/
├── models/
├── notebooks/
│   └── 01_EDA.ipynb
├── src/
│   ├── anomaly_detection.py
│   ├── evaluate.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/AAROHI-AGARWAL/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add the dataset

Download `creditcard.csv` from the dataset source above and place it in:

```text
data/raw/creditcard.csv
```

Ensure the cleaned dataset is available at:

```text
data/processed/creditcard_cleaned.csv
```

### 5. Train the models

From the project root, run:

```bash
python src/train.py
```

This generates the trained model files and thresholds in the `models/` folder.

### 6. Evaluate the models

```bash
python src/evaluate.py
```

### 7. Run the FastAPI application

```bash
uvicorn api.main:app --reload
```

Open the interactive API documentation at:

http://127.0.0.1:8000/docs

### 8. Run the Streamlit dashboard

Open a second terminal from the project root, activate the virtual environment, and run:

```bash
streamlit run dashboard/app.py
```

## Important Notes

- The dataset, trained model binaries, and local SQLite database are excluded from version control.
- Train the models before running the API or dashboard.
- The model uses anonymized dataset features; it does not establish whether a real-world transaction is fraudulent.
- Predictions should be treated as risk indicators for further review, not definitive proof of fraud.

## Future Improvements

- Experiment with additional anomaly detection algorithms.
- Improve recall while managing false positives.
- Add authentication and stronger API validation.
- Add automated tests and Docker support.
- Explore deployment to a cloud platform.

## Author

Developed as an academic machine learning project.
