# IPL Match Prediction with Machine Learning and ANN
* This project leverages machine learning to predict the outcome of IPL (Indian Premier League) matches based on various features such as team performance, match statistics, and in-game metrics. The model is trained on historical IPL data and includes a comprehensive analysis pipeline using multiple algorithms and advanced techniques.
![image](https://github.com/user-attachments/assets/30282ce6-e895-4958-b7ca-447ce8429aa6)

### Overview
* The goal of this project is to provide accurate predictions on the outcome of IPL matches, leveraging the power of machine learning and MLOps to build, deploy, and maintain the model. The project includes the following stages:

* **Data Collection:** Historical data on IPL matches, including runs, wickets, city, and match outcomes.
* **Data Preprocessing:** Cleaning, feature engineering, and data transformation for model building.
* **Model Training:** Training models such as Random Forest, XGBoost, and deep learning-based neural networks.
* **Model Evaluation:** valuating models on accuracy, F1 score, and other metrics.
* **MLOps:**  Using tools like MLflow for experiment tracking and versioning of models, ensuring the smooth deployment and monitoring of the models.

### Project Features
* **Data Analysis:** Performed comprehensive analysis to identify patterns and trends that influence match outcomes.
* **Multiple Algorithms:** The project compares various machine learning models such as Random Forest, XGBoost, and neural networks to determine the best performing model.
* **Model Optimization:** Hyperparameter tuning with GridSearchCV to optimize model performance.
* **MLOps Integration:** Using MLflow to log experiments, monitor model performance, and manage the lifecycle of machine learning models.
* **Deployment:** Models are saved, versioned, and can be deployed for real-time predictions.

### Tools & Technologies Used
* **Hugging Face:** used for creating streamlit app 
* **MLflow:** For managing experiments and tracking model metrics.
* **Scikit-learn:** For building machine learning models and performing cross-validation.
* **XGBoost:** For high-performing gradient boosting models.
* **TensorFlow/Keras:** For deep learning model development.
* **Pandas:** For data processing and manipulation.


## Analysis
* Data exploration and feature engineering revealed important patterns such as match conditions (city), team performance, and wickets taken, which were crucial for improving model accuracy.
* Various algorithms were tested, and Random Forest emerged as the best performing model due to its ability to handle large datasets with varying feature importance.
* ![image](https://github.com/user-attachments/assets/0006fa64-5b1a-4176-8353-dee35d861d95)


# Conclusion
* This project demonstrates the power of machine learning and MLOps in real-world applications like IPL match prediction. The integration of Hugging Face and MLflow allows for efficient tracking, versioning, and deployment of machine learning models.
