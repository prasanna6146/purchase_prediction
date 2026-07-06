# Online Purchase Prediction

A simple machine learning script that predicts whether a customer will make an online purchase based on their age, salary, and website browsing behavior. It uses **Logistic Regression** from scikit-learn.

## How It Works

1. Loads customer data from a CSV file
2. Trains a Logistic Regression model on historical purchase data
3. Evaluates the model's accuracy on a test split
4. Prompts you to enter a new customer's details
5. Predicts whether that customer will purchase

## Features Used

| Feature | Description |
|---|---|
| `Age` | Customer's age |
| `Salary` | Customer's salary |
| `Time_Spent_On_Website` | Minutes spent browsing the site |
| `Pages_Visited` | Number of pages viewed |

**Target:** `Purchased` (1 = purchased, 0 = did not purchase)

## Requirements

- Python 3.7+
- pandas
- scikit-learn

Install dependencies:

```bash
pip install pandas scikit-learn
```

## Dataset

Place a file named `online_purchase.csv` in the same directory as the script. It must contain the following columns:

```
Age, Salary, Time_Spent_On_Website, Pages_Visited, Purchased
```

> **Note:** This dataset is not included in this repository. You'll need to supply your own data with the columns listed above.

## Usage

Run the script from the command line:

```bash
python purchase_prediction.py
```

The script will:
1. Print a preview of the loaded dataset
2. Print the model's accuracy on the test set
3. Ask you to input details for a new customer:
   - Age
   - Salary
   - Time Spent On Website (minutes)
   - Pages Visited
4. Output whether the customer is predicted to purchase

### Example

```
Enter Customer Details
Age: 34
Salary: 55000
Time Spent On Website (minutes): 12
Pages Visited: 8

Prediction Result
Customer Will Purchase
```

## Notes & Possible Improvements

- The model uses a fixed `random_state=42` and an 80/20 train-test split for reproducibility.
- No feature scaling is currently applied — adding `StandardScaler` could improve model performance for logistic regression.
- Consider adding input validation for the interactive prompts (e.g., handling non-numeric input).
- Consider saving/loading the trained model (e.g., with `joblib`) to avoid retraining on every run.

## License

Add your preferred license here (e.g., MIT).
