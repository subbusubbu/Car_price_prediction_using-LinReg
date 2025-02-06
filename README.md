# Car_price_prediction_using-LinReg
Predicted car price using Linear Regression model which showed 99%accuracy

### **README File for Car Price Prediction Using Linear Regression**

---

# **Car Price Prediction Using Linear Regression 🚗💰**
This project aims to predict the price of used cars based on various features such as brand, model, year, engine size, mileage, fuel type, and transmission. The model is built using **Multiple Linear Regression**, and the dataset is sourced from **Kaggle**.

## 📌 **Project Overview**
- **Dataset:** Car Price Dataset from Kaggle.
- **Goal:** Predict car prices using historical data and key car attributes.
- **Tech Stack:** Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn.
- **Algorithm Used:** Multiple Linear Regression.

---

## 📊 **Dataset Description**
The dataset consists of 10,000 rows with the following features:

| Feature         | Description |
|----------------|------------|
| `Brand`        | Car brand (e.g., Volkswagen, Toyota) |
| `Model`        | Specific car model |
| `Year`         | Manufacturing year |
| `Engine_Size`  | Engine size in liters |
| `Fuel_Type`    | Type of fuel (Diesel, Petrol, Electric, Hybrid) |
| `Transmission` | Transmission type (Manual, Automatic, Semi-Automatic) |
| `Mileage`      | Distance covered (in km) |
| `Doors`        | Number of doors |
| `Owner_Count`  | Number of previous owners |
| `Price`        | Target variable (Car Price in USD) |

---

## 🔧 **Project Workflow**
1. **Data Loading & Exploration**
   - Load dataset using Pandas.
   - Check for missing values and data types.
   - Perform descriptive statistics.

2. **Feature Engineering**
   - Handle categorical variables using **One-Hot Encoding (OHE)**.
   - Drop unnecessary columns.
   - Split data into training and testing sets.

3. **Model Training**
   - Train a **Linear Regression Model** on the training set.
   - Fit the model to predict car prices.

4. **Model Evaluation**
   - Compute **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **Root Mean Squared Error (RMSE)**.
   - Calculate **R² score** (Coefficient of Determination).

---

## 📈 **Model Performance**
- **Mean Absolute Error (MAE):** `20.00`
- **Mean Squared Error (MSE):** `4213.92`
- **Root Mean Squared Error (RMSE):** `64.91`
- **R² Score:** `0.9995` (99% Accuracy)

🚀 The model performs exceptionally well, achieving **99% accuracy** in predicting car prices!

---

## 📦 **Installation & Usage**
### **Prerequisites**
- Python 3.x
- Jupyter Notebook / Google Colab
- Required Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `kagglehub`

### **Steps to Run the Project**
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/car-price-prediction.git
   cd car-price-prediction
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
   ```
3. Download the dataset using KaggleHub:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("asinow/car-price-dataset")
   ```
4. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook Car_Price_Prediction.ipynb
   ```
5. The model will be trained and tested automatically.


🚀 **Happy Coding!** 🚀
