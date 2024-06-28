# Movie-Genre-Classification-ML
🎥📊 Movie Genre Classification with Machine Learning

## 📝 Description:
This project aims to classify the genre of movies based on their plot summaries or other textual information using machine learning techniques. It utilizes natural language processing (NLP) techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) vectorization combined with various classifiers including Linear Support Vector Machines (LinearSVC), Multinomial Naive Bayes (MultinomialNB), and Logistic Regression.

## 🔍 Features:
Data Exploration: Visualizes the distribution of movie genres, description lengths, and the top 10 most frequent genres in the training data.
Text Vectorization: Converts textual descriptions into TF-IDF features for machine learning model input.
Model Training: Utilizes multiple classifiers including LinearSVC, MultinomialNB, and Logistic Regression for genre prediction.
Model Evaluation: Evaluates model performance using accuracy scores and classification reports.
Prediction Functionality: Implements a function to predict movie genres based on provided descriptions.

## 🚀 Usage:
Clone the repository.
Install the required libraries (scikit-learn, pandas, numpy, seaborn, matplotlib) if not already installed.
Execute the Python script to explore the data, train the models, and make predictions.

## 📄 Data Files:
train_data.txt: Training data containing movie IDs, titles, genres, and descriptions.
test_data.txt: Test data containing movie IDs, titles, genres, and descriptions for evaluation.
test_data_solution.txt: Solution data containing movie IDs, titles, genres, and descriptions for evaluating model performance.

## 🔗 Dependencies:
Python 3.x
scikit-learn
pandas
numpy
seaborn

## 📈 Results:
The project achieves successful genre classification with various machine learning models. Model performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.
