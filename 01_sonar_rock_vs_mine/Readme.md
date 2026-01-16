# Sonar Rock vs Mine Classification using Logistic Regression

## Project Overview

This mini project implements a **binary classification system** to distinguish between **rocks** and **mines** using **Logistic Regression from scikit-learn**. The model is trained on the well-known **Sonar dataset**, which contains numerical features extracted from sonar signals bounced off different objects.

The goal of this project is to practice a complete **machine learning workflow** using standard Python ML tools:

* Data loading and exploration
* Train–test split
* Model training with scikit-learn
* Model evaluation using accuracy
* Making predictions on new input data

---

## Project Structure

```
01_sonar_rock_vs_mine/
│
├── main.py              # Main script: data processing, training, evaluation, prediction
├── sonar_data.csv       # Sonar dataset (features + target label)
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

---

## Dataset Description

* The dataset contains **61 columns**:

  * Columns `0` to `59`: numerical sonar signal features
  * Column `60`: target label

    * `R` → Rock
    * `M` → Mine

* The dataset is loaded using **pandas** and analyzed with basic statistics and class distribution checks.

---

## Model Used

* **Algorithm**: Logistic Regression
* **Library**: `scikit-learn`
* **Why Logistic Regression?**

  * Suitable for binary classification
  * Simple, interpretable baseline model
  * Efficient for small to medium-sized datasets

---

## Workflow

1. Import required libraries
2. Load and inspect the dataset
3. Separate features (`X`) and labels (`Y`)
4. Split data into training and test sets using stratification
5. Train a Logistic Regression model on training data
6. Evaluate accuracy on:

   * Training set
   * Test set
7. Build a simple predictive system for a single sonar input

---

## Usage

1. Clone the repository or download the project folder
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:

   ```bash
   python main.py
   ```

The script will:

* Display dataset information
* Train the model
* Print training and test accuracy
* Predict whether a given sonar signal corresponds to a **Rock** or a **Mine**

---

## Example Prediction Output

```
['R']
The object is Rock
```

---

## Requirements

* Python 3.x
* numpy
* pandas
* scikit-learn

(Exact versions can be found in `requirements.txt`.)

---

## Learning Outcomes

Through this mini project, you will gain hands-on experience with:

* Practical use of scikit-learn
* Binary classification problems
* Data preprocessing and evaluation
* End-to-end ML project structure

---

## Contributions

Contributions are welcome.
If you find a bug, want to optimize the code, or improve the documentation, feel free to:

* Open an issue
* Submit a pull request

---

## Author

**Melik Fourati**

Mini Machine Learning Project
