# 🧑‍💼 Employee Attrition Prediction System

A Machine Learning-powered web application built with **Streamlit** that predicts whether an employee is likely to leave an organization based on HR-related attributes. This project compares multiple machine learning models and provides an interactive dashboard for real-time employee attrition prediction.

---

# 🌐 Live Demo

### 🚀 Streamlit Application

🔗 **Live App:**  
https://employeeattritionprediction-fqtitlambuqdw7qvpaqztf.streamlit.app/

### 💻 GitHub Repository

🔗 **GitHub:**  
https://github.com/amiryousra989-web/Employee_Attrition_Prediction

---

# 📂 Project Structure

```text
Employee_Attrition_Prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   └── Employee-Attrition.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── model_columns.pkl
│
├── images/
│   ├── model_comparison.png
│   ├── confusion_matrix_logistic.png
│   ├── confusion_matrix_decision_tree.png
│   └── confusion_matrix_random_forest.png
│
└── Notebooks/
    └── Employee_Attrition.ipynb
```

---

# 📖 Project Overview

Employee attrition is a significant challenge for organizations, affecting productivity, recruitment costs, and overall business performance.

This project leverages Machine Learning techniques to analyze employee-related factors and predict whether an employee is likely to leave the company. The application provides HR professionals with valuable insights that can support employee retention strategies.

---

# ✨ Features

- 🧑‍💼 Employee Attrition Prediction
- 🤖 Multiple Machine Learning Models
- 📊 Model Performance Comparison
- 📈 Probability-based Prediction
- 🎨 Modern Dark-Themed User Interface
- ⚡ Real-Time Predictions
- 🌐 Interactive Streamlit Web Application
- 📉 Visual Model Evaluation

---

# 🤖 Machine Learning Models

The following machine learning algorithms were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

### 🏆 Best Performing Model

**Logistic Regression**

---

# 📊 Model Performance

| Model | Accuracy |
|--------|----------|
| **Logistic Regression** | **86.05%** |
| Random Forest | 83.33% |
| Decision Tree | 76.53% |

Logistic Regression achieved the highest overall accuracy and demonstrated the best balance between predictive performance and model simplicity.

---

# 🧠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

---

# 📁 Dataset

**Dataset Used**

IBM HR Analytics Employee Attrition & Performance Dataset

The dataset contains employee demographic information, job-related features, compensation details, work environment metrics, and performance indicators used to predict employee attrition.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/amiryousra989-web/Employee_Attrition_Prediction.git
```

## 2. Navigate to the Project Folder

```bash
cd Employee_Attrition_Prediction
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Application

```bash
streamlit run app.py
```

---

# 🔮 How to Use

1. Launch the Streamlit application.
2. Enter the employee information.
3. Select the desired Machine Learning model.
4. Click **Predict Attrition**.
5. View the prediction result and probability of employee attrition.

---

# 📈 Future Improvements

- XGBoost & LightGBM Models
- Deep Learning Implementation
- SHAP Explainability
- Employee Dashboard Analytics
- Cloud Database Integration
- Authentication System
- Downloadable Prediction Reports
- REST API Integration

---



# 🎯 Project Objectives

- Predict employee attrition using Machine Learning.
- Compare multiple classification algorithms.
- Build an interactive web application for HR analytics.
- Demonstrate an end-to-end Machine Learning workflow from data preprocessing to deployment.

---

# 👩‍💻 Author

**Yousra Amir**

📧 Email: amiryousra989@gmail.com

🐙 GitHub:  
https://github.com/amiryousra989-web

🌐 Live Application:  
https://employeeattritionprediction-fqtitlambuqdw7qvpaqztf.streamlit.app/

---

# ⚠️ Disclaimer

This project has been developed for **educational and portfolio purposes**. Predictions are generated using historical HR data and should be used as decision-support tools rather than as the sole basis for employee retention decisions.

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub!

**GitHub Repository:**  
https://github.com/amiryousra989-web/Employee_Attrition_Prediction
