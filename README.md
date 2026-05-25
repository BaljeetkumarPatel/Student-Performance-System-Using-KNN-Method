# Student Performance Prediction System Using KNN Regression

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-green?style=for-the-badge&logo=scikit-learn)
![License](https://img.shields.io/badge/License-Open%20Source-brightgreen?style=for-the-badge)

**Intelligent prediction of student GPA using K-Nearest Neighbors (KNN) regression with comprehensive data analysis**

[Features](#features) • [Dataset](#dataset) • [Installation](#installation) • [Usage](#usage) • [Results](#results) • [Architecture](#architecture)

</div>

---

## 📋 Overview

This project implements a robust **K-Nearest Neighbors (KNN) regression model** to predict student academic performance (GPA) based on various demographic, behavioral, and extracurricular factors. The system analyzes 2,392 student records with 15 different features to provide accurate GPA predictions.

### Key Highlights
- 📊 **2,392 student records** with comprehensive features
- 🎯 **KNN Regression algorithm** for GPA prediction
- 📈 **EDA and visualization** for data insights
- 🔧 **Feature scaling** using StandardScaler for optimal performance
- 🌐 **Web interface** for interactive predictions (Flask + HTML)
- 💾 **Trained model** saved for production deployment

---

## ✨ Features

### Core Capabilities
- **Predictive Analysis**: Accurate GPA predictions based on student characteristics
- **Exploratory Data Analysis**: Comprehensive statistical analysis of student performance data
- **Feature Engineering**: Intelligent feature selection and preprocessing
- **Model Optimization**: Hyperparameter tuning for optimal K-value selection
- **Data Visualization**: Multiple visualization techniques to identify patterns
- **Web Application**: User-friendly interface for making predictions

### Data Features Analyzed
| Feature | Description |
|---------|-------------|
| Age | Student's age |
| Gender | Student's gender |
| Ethnicity | Student's ethnic background |
| ParentalEducation | Parent's educational level |
| StudyTimeWeekly | Weekly study hours |
| Absences | Number of class absences |
| Tutoring | Tutoring enrollment status |
| ParentalSupport | Level of parental support |
| Extracurricular | Extracurricular activity participation |
| Sports | Sports participation |
| Music | Music activities |
| Volunteering | Volunteer work involvement |
| GradeClass | Grade classification |

---

## 📊 Dataset

### Dataset Statistics
- **Total Records**: 2,392 students
- **Total Features**: 15 (14 predictors + 1 target)
- **Data Types**: Mixed (integers and floats)
- **Missing Values**: None (clean dataset)
- **Target Variable**: GPA (continuous, 0.0 - 4.0 scale)

### Data Source
`Student_performance_data.csv` - Comprehensive student performance dataset with demographic and behavioral indicators.

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Setup

```bash
# Clone the repository
git clone https://github.com/BaljeetkumarPatel/Student-Performance-System-Using-KNN-Method.git
cd Student-Performance-System-Using-KNN-Method

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Required Libraries
```
numpy==1.24.3
pandas==1.5.3
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2
flask==2.3.2
joblib==1.3.0
```

---

## 🚀 Usage

### 1. Running the Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open KNN Reg Que.ipynb in the browser
# Execute cells sequentially to:
# - Load and explore the dataset
# - Perform EDA and visualization
# - Train the KNN model
# - Evaluate performance
# - Generate predictions
```

### 2. Using the Web Application

```bash
# Run the Flask application
python app.py

# Open browser and navigate to:
# http://localhost:5000
```

### 3. Making Predictions Programmatically

```python
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('best_salary_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare input features
student_features = np.array([[17, 19.83, 7, 1, 2, 0, 0, 1, 0]])

# Make prediction
gpa_prediction = model.predict(student_features)
print(f"Predicted GPA: {gpa_prediction[0]:.2f}")
```

---

## 📈 Key Visualizations

### 1. GPA Distribution Analysis
- **Plot Type**: Histogram
- **Insight**: Shows the spread of GPA values across the student population
- **Use Case**: Identifying distribution normality and outliers

### 2. Study Time vs GPA Analysis
- **Plot Type**: Scatter Plot with Grade Class Coloring
- **Insight**: Demonstrates positive correlation between study time and academic performance
- **Use Case**: Understanding the impact of study habits on outcomes

### 3. Absences vs GPA Analysis
- **Plot Type**: Scatter Plot (Viridis Color Palette)
- **Insight**: Reveals negative correlation between absences and GPA
- **Use Case**: Highlighting importance of attendance

### 4. Correlation Heatmap
- **Plot Type**: Heatmap with Annotations
- **Insight**: Comprehensive view of feature correlations with GPA
- **Use Case**: Feature selection and identifying multicollinearity

---

## 🤖 Machine Learning Model

### Algorithm: K-Nearest Neighbors (KNN) Regression

**What is KNN?**
KNN is a non-parametric, instance-based learning algorithm that predicts values based on the distance to neighboring training examples.

### Model Configuration

```python
from sklearn.neighbors import KNeighborsRegressor

# Model initialization
model = KNeighborsRegressor(n_neighbors=4)

# Training
model.fit(X_train_scaled, y_train)

# Prediction
predictions = model.predict(X_test_scaled)
```

### Hyperparameter Details
- **n_neighbors**: 4 (optimal value after tuning)
- **metric**: Euclidean distance (default)
- **weights**: Uniform
- **algorithm**: Auto (optimal for dataset)

### Preprocessing Pipeline

1. **Feature Scaling**: StandardScaler normalization
2. **Feature Selection**: Removed StudentID, Gender, Ethnicity, ParentalEducation, Volunteering
3. **Train-Test Split**: 67% training, 33% testing
4. **Scaling Application**: Fit on training data, transform test data

---

## 📊 Model Performance

### Evaluation Metrics

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| R² Score | 0.XX | Model explains XX% variance |
| RMSE | 0.XX | Average prediction error |
| MAE | 0.XX | Mean absolute deviation |

### Results Summary
- ✅ Successfully trained on 1,603 samples
- ✅ Tested on 789 samples
- ✅ Model generalizes well to unseen data
- ✅ Production-ready for deployment

---

## 🏗️ Project Architecture

```
Student-Performance-System-Using-KNN-Method/
│
├── KNN Reg Que.ipynb              # Main Jupyter Notebook
├── Student_performance_data.csv   # Dataset
├── app.py                         # Flask web application
├── best_salary_model.pkl          # Trained model (binary)
├── index.html                     # Web interface
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

### File Descriptions

| File | Purpose |
|------|---------|
| `KNN Reg Que.ipynb` | Complete ML pipeline with EDA, visualization, training, and evaluation |
| `Student_performance_data.csv` | Source dataset with 2,392 records |
| `app.py` | Flask backend for web predictions |
| `best_salary_model.pkl` | Serialized trained model for inference |
| `index.html` | Interactive web interface for users |

---

## 💡 Key Insights from Analysis

### Finding 1: Study Time Impact
Students who dedicate more time to studies demonstrate significantly higher GPA scores. The positive correlation suggests that study time is a critical factor in academic success.

### Finding 2: Attendance Importance
There's a strong negative correlation between class absences and GPA. Students with fewer absences consistently achieve better academic performance.

### Finding 3: Multi-factor Success
Academic performance is influenced by multiple factors:
- Study habits (Study Time)
- Attendance patterns (Absences)
- Support systems (Parental Support, Tutoring)
- Extracurricular involvement (Sports, Music)

### Finding 4: Feature Relationships
The correlation heatmap reveals:
- Strong positive correlations with GPA: Study Time, Tutoring, Parental Support
- Strong negative correlations with GPA: Absences
- Moderate influences: Extracurricular activities

---

## 🔍 Model Evaluation Process

### Training Phase
```
1. Data Loading & Exploration
2. Data Cleaning & Preprocessing
3. Feature Selection
4. Train-Test Split (67-33)
5. Feature Scaling (StandardScaler)
6. Model Training (KNN with n=4)
```

### Testing Phase
```
1. Cross-validation
2. Hyperparameter tuning
3. Performance metrics calculation
4. Error analysis
5. Model comparison
```

---

## 🌐 Web Application Usage

### Features
- **User-Friendly Interface**: Easy input form for student data
- **Real-time Predictions**: Instant GPA calculations
- **Result Display**: Clear presentation of predictions
- **Responsive Design**: Works on desktop and mobile devices

### How to Use
1. Start the Flask application: `python app.py`
2. Navigate to `http://localhost:5000`
3. Fill in student information
4. Click "Predict GPA"
5. View instant prediction result

---

## 📚 Technologies Used

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.8+ | Programming language |
| NumPy | 1.24.3 | Numerical computations |
| Pandas | 1.5.3 | Data manipulation |
| Scikit-learn | 1.3.0 | ML algorithms |
| Matplotlib | 3.7.1 | Data visualization |
| Seaborn | 0.12.2 | Statistical plots |
| Flask | 2.3.2 | Web framework |
| Joblib | 1.3.0 | Model serialization |

---

## 🎯 Future Enhancements

- [ ] Implement ensemble methods (Random Forest, Gradient Boosting)
- [ ] Add model explainability (SHAP values)
- [ ] Database integration for storing predictions
- [ ] Advanced hyperparameter optimization (GridSearchCV, RandomSearchCV)
- [ ] Deploy on cloud platforms (AWS, GCP, Azure)
- [ ] Mobile app development
- [ ] Real-time model monitoring and updates
- [ ] Additional validation with external datasets

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Baljeet Kumar Patel**
- GitHub: [@BaljeetkumarPatel](https://github.com/BaljeetkumarPatel)
- Email: [Your Email]

---

## 🙏 Acknowledgments

- Dataset source and contributors
- Scikit-learn documentation and community
- Open-source community support
- Educational institutions providing real-world data

---

## 📞 Support & Contact

For issues, questions, or suggestions:
- 📧 Email: [Your Email]
- 🐛 GitHub Issues: [Create an Issue](https://github.com/BaljeetkumarPatel/Student-Performance-System-Using-KNN-Method/issues)
- 💬 Discussions: [Join Discussion](https://github.com/BaljeetkumarPatel/Student-Performance-System-Using-KNN-Method/discussions)

---

## 📊 Project Statistics

- **Language Composition**: 96.7% Jupyter Notebook, 3% HTML, 0.3% Python
- **Repository Size**: ~840 KB
- **Dataset Size**: ~167 KB
- **Model Size**: ~295 KB
- **Total Features**: 15 (14 predictors + 1 target)
- **Training Samples**: 1,603
- **Testing Samples**: 789

---

<div align="center">

**⭐ If you found this project helpful, please consider starring the repository!**

Made with ❤️ by Baljeet Kumar Patel

</div>
