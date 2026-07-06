import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("online_purchase.csv")

print("Dataset Loaded Successfully")
print(df.head())

# Features
X = df[
    [
        "Age",
        "Salary",
        "Time_Spent_On_Website",
        "Pages_Visited"
    ]
]

# Target
y = df["Purchased"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)

# Accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# User Inputs
print("\nEnter Customer Details")

age = int(input("Age: "))
salary = int(input("Salary: "))
time_spent = int(
    input("Time Spent On Website (minutes): ")
)

pages = int(
    input("Pages Visited: ")
)

prediction = model.predict(
    [[
        age,
        salary,
        time_spent,
        pages
    ]]
)

print("\nPrediction Result")

if prediction[0] == 1:
    print("Customer Will Purchase")
else:
    print("Customer Will NOT Purchase")