import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from algorithms import LinearRegression

def main():
    # Importing dataset
    file_path = os.path.join(os.path.dirname(__file__), "salary_data.csv")
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, 1].values

    # Splitting dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=1 / 3, random_state=0)

    # Model training
    model = LinearRegression(iterations=1000, learning_rate=0.01)
    model.fit(X_train, Y_train)

    # Prediction
    Y_pred = model.predict(X_test)

    print("Predicted values ", np.round(Y_pred[:3], 2))
    print("Real values      ", Y_test[:3])
    print("Trained W        ", round(model.W[0], 2))
    print("Trained b        ", round(model.b, 2))

    # Visualization
    plt.scatter(X_test, Y_test, color='blue')
    plt.plot(X_test, Y_pred, color='orange')
    plt.title('Salary vs Experience')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()


if __name__ == "__main__":
    main()
