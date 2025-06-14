
# 🫀 Heart Disease Classification using Machine Learning

This project implements a **Heart Disease Prediction System** using multiple machine learning algorithms to classify the presence of heart disease based on various clinical features.

## 📌 Table of Contents

- [About](#about)
- [Tools & Technologies Used](#tools--technologies-used)
- [Dataset](#dataset)
- [Features](#features)
- [ML Algorithms Used](#ml-algorithms-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 📖 About

The goal of this project is to use machine learning to predict the presence of heart disease based on medical attributes. It uses the UCI Heart Disease dataset and tests the performance of five popular classification algorithms.

---

## 🛠️ Tools & Technologies Used

| Tool/Library       | Purpose                      |
|--------------------|------------------------------|
| Python             | Programming Language         |
| Pandas             | Data manipulation            |
| NumPy              | Numerical operations         |
| Scikit-learn       | ML models and preprocessing  |
| Matplotlib & Seaborn | Data visualization        |
| Jupyter Notebook / VS Code | Development IDE    |

---

## 📁 Dataset

Dataset Source:  
- UCI ML Repository: [Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)  
- Kaggle: [Heart Disease UCI](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)

You can also use the `heart.csv` file provided in this repository.

---

## ✅ Features

The dataset contains the following features:

- `age`
- `sex`
- `cp` (chest pain type)
- `trestbps` (resting blood pressure)
- `chol` (serum cholesterol)
- `fbs` (fasting blood sugar)
- `restecg` (resting ECG)
- `thalach` (maximum heart rate)
- `exang` (exercise-induced angina)
- `oldpeak` (ST depression)
- `slope` (slope of the ST segment)
- `ca` (number of major vessels)
- `thal`
- `target` (0 = no disease, 1 = has disease)

---

## 🧠 ML Algorithms Used

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier

---

## 💻 Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/heart-disease-classification.git
cd heart-disease-classification

# Install required libraries
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 🚀 Usage

1. Place `heart.csv` in the project directory.
2. Run the script:
```bash
python heart_disease_prediction.py
```
3. View the accuracy, classification report, and feature importance chart in the output.

---

## 📊 Results

Example model performance (accuracy may vary by dataset size):

| Model               | Accuracy (Sample) |
|--------------------|-------------------|
| Logistic Regression | ~85%             |
| Random Forest       | ~90%             |
| SVM                 | ~86%             |
| KNN                 | ~84%             |
| Decision Tree       | ~80%             |

---

## 🔮 Future Improvements

- Hyperparameter tuning using GridSearchCV
- K-Fold Cross-validation
- Model deployment using Flask or Streamlit
- Ensemble learning with VotingClassifier
- Larger real-world dataset

---

## 📜 License

This project is open-source and available under the MIT License.
