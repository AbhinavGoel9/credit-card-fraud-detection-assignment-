# Credit Card Fraud Detection  

This project builds a machine learning model to detect fraudulent credit card transactions based on transaction data.  

## Dataset  
The dataset file (`creditcard.csv`) exceeds GitHub's upload limit (25MB) and cannot be included in this repository.  
To use this project, please **download the dataset manually** from the following link:  

🔗 [Credit Card Fraud Detection Dataset - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  

Once downloaded, **place `creditcard.csv` in the same folder as the Python script** before running it.  

## Requirements  
Before running the project, install the required dependencies:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## How to Run  
1. **Download the dataset** from Kaggle and save it as `creditcard.csv` in the project folder.  
2. **Run the script**:  
   ```bash
   python credit_card_fraud.py
   ```

## Model and Approach  
- **Feature Engineering:**  
  - Transaction frequency per user  
  - Spending pattern analysis  
  - High-value transaction detection  
- **Handling Class Imbalance:**  
  - Applied **SMOTE (oversampling)**  
  - Applied **Random Undersampling**  
- **Model Selection & Evaluation:**  
  - Trained and compared **Random Forest & Logistic Regression**  
  - Evaluated using **Accuracy, Precision, Recall, and F1-Score**  
  - Visualized results using **Confusion Matrix Heatmaps**  

## Output  
The script prints key evaluation metrics and generates confusion matrix heatmaps for both models.  
