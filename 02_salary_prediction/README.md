# 02 - Salary Prediction (Linear Regression from Scratch)

A clean, educational implementation of **Linear Regression from Scratch** using only NumPy  
Predicting salary based on years of experience

<p align="center">
  <img src="https://img.shields.io/badge/Linear%20Regression-From%20Scratch-blue?style=for-the-badge&logo=python&logoColor=white" alt="From Scratch">
  <img src="https://img.shields.io/badge/NumPy-Powered-orange?style=for-the-badge&logo=numpy" alt="NumPy">
</p>

## ✨ Project Highlights

- Pure **NumPy** implementation of Linear Regression (no scikit-learn model)
- Clean separation between algorithm and usage code
- Visual comparison: scratch implementation vs scikit-learn
- Simple, well-commented educational code
- Professional-looking result visualization

## 📂 Project Structure

```text
02_Salary_Prediction/
├── main.py               # Demo + training + visualization
├── salary_data.csv       # Training dataset
└── README.md

↳ algorithms/LinearRegression.py   # Core algorithm (shared across projects)
```


---

## Dataset Description

- **File**: `salary_data.csv`
- **Columns**:
  - `YearsExperience` → number of years (independent variable)
  - `Salary` → monthly/yearly salary in $ (target variable)
- Small classic dataset (~30 rows) — perfect for learning and visualization

---

## Model Implemented

**Algorithm**: Linear Regression (from scratch)  
**Core technique**: Gradient Descent  
**Only external library used for math**: NumPy

**Model equation**:
Salary = W × YearsExperience + b

Where:
- `W` → weight/slope
- `b` → bias/intercept

---

## Workflow

1. Load the dataset using pandas
2. Split data into training and test sets (1/3 test size)
3. Train custom `LinearRegression` model using gradient descent
4. Make predictions on test set
5. Show first few predictions vs real values
6. Display learned parameters (W and b)
7. Visualize: scatter plot + regression line

---

## Usage


1. Make sure you're in the project folder
   ```bash
   cd 02_salary_prediction  
    ```
2. Install dependencies:

   ```bash
   pip install numpy pandas matplotlib
   ```
3. Run the main script:

   ```bash
   python main.py
   ```

