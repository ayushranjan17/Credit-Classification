# Credit-Classification

## Project Overview
This project focuses on classifying bank customers based on their credit risk using the German Credit Dataset. The primary objective is to develop and evaluate machine learning models to predict whether a customer is a good or bad credit risk. Given the imbalanced nature of the dataset, special emphasis is placed on minimizing misclassification errors, especially false negatives, which are more costly for banks.

## Dataset
The German Credit Dataset consists of 1,000 customer records with 20 input variables:
- **Numerical Variables (7)**: Duration in months, Credit amount, Age, etc.
- **Categorical Variables (13)**: Checking account status, Credit history, Employment status, etc.

The target variable has two classes:
- **Good Customers (70%)**: Majority class (negative class)
- **Bad Customers (30%)**: Minority class (positive class)

## Project Workflow
1. **Data Exploration**
   - Load and visualize the dataset
   - Check class distribution and variable types
   - Apply necessary preprocessing (one-hot encoding, scaling)
2. **Baseline Model**
   - Implement a dummy classifier
   - Establish a baseline F2-score for comparison
3. **Model Evaluation**
   - Test machine learning models: Logistic Regression, LDA, Naive Bayes, SVM, etc.
   - Use stratified k-fold cross-validation with F2-score metric
   - Optimize performance using undersampling techniques (ENN, RENN, NCR, etc.)
4. **Final Model Selection & Prediction**
   - Select the best-performing model based on F2-score
   - Train on the full dataset and make predictions on new customer data

## Technologies Used
- **Python**
- **Libraries**:
  - `pandas`, `numpy` for data manipulation
  - `scikit-learn` for machine learning models
  - `imbalanced-learn` for handling class imbalance
  - `matplotlib` for visualization

## Installation & Usage
### Prerequisites
Ensure you have Python 3.x installed along with the required libraries:
```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib
```

### Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/german-credit-classification.git
   cd german-credit-classification
   ```
2. Download the dataset:
   ```bash
   wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv
   ```
3. Run the main script:
   ```bash
   python main.py
   ```

## Results
- The best model achieved an F2-score of ~0.716 using Logistic Regression with Repeated ENN undersampling.
- The model correctly identifies high-risk customers while minimizing false negatives.

## Future Improvements
- Implement hyperparameter tuning for better model performance.
- Explore advanced ensemble methods and deep learning approaches.
- Use cost-sensitive learning techniques to further reduce financial risk.

## References
- [German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data))
- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [Imbalanced-Learn Documentation](https://imbalanced-learn.readthedocs.io/en/stable/)
