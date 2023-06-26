
# Sentiment Analysis from US Airline Customers' Twitter
This data science project focuses on conducting sentiment analysis on tweets from US airline customers. The main objective is to analyze customer sentiment and compare the performance of machine learning algorithms such as Naive Bayes, Support Vector Machine (SVM), Logistic Regression, and Random Forest. Additionally, balancing techniques including undersampling, oversampling, and SMOTE are utilized to address the class imbalance in the dataset and improve prediction accuracy.


## Objective
The objective of this project is to perform sentiment analysis on US airline customer tweets and evaluate the effectiveness of different machine learning algorithms and balancing techniques in accurately predicting customer sentiment. By comparing the results obtained from various models, we aim to identify the most effective approach for sentiment analysis in this context.
## Dataset Used
The dataset used in this project consists of tweets from US airline customers. It contains 14,640 rows and 15 columns, including features such as the tweet text and the corresponding sentiment (negative, neutral, or positive). The dataset provides a real-world representation of customer opinions and sentiments regarding their airline experiences.
## Prerequisites
To run this project, ensure you have the following dependencies installed:

pandas

nltk

scikit-learn

imbalanced-learn
## Methodology
1. Loading the dataset: The dataset is loaded, containing tweets and corresponding sentiment labels.
2.  Data Preparation:
-   Cleaning text data: Special characters are removed, and the text is converted to lowercase.
-   Stemming and stop word removal: Words are stemmed using the Porter Stemmer algorithm, and common stop words are removed.
-   Vectorization: The cleaned text is transformed into numerical feature vectors using the CountVectorizer.
3.  Model Training:
-   Naive Bayes: The Naive Bayes classifier is trained on the preprocessed data, and predictions are made.
-  SVM: Support Vector Machine model is trained and predictions are made.
-   Logistic Regression: Logistic Regression model is trained and predictions are made.
-   Random Forest: Random Forest Classifier is trained and predictions are made.
4.  Balancing Techniques:
-   Undersampling: The dataset is balanced by randomly selecting a subset of instances from the majority class.
-   Oversampling: The minority class is oversampled to achieve a balanced dataset.
-   SMOTE: Synthetic Minority Over-sampling Technique is employed to generate synthetic samples of the minority class.
5.  Model Evaluation:
-   The trained models are evaluated using performance metrics such as accuracy, precision, recall, and F1 score. Classification reports are generated for each model to assess their performance on different sentiment classes.
## Results
By evaluating multiple ML models, we found that the Gradient Boosting Classifier, Support Vector Classifier (SVC), and AdaBoost Classifier demonstrated higher recall scores. These models exhibit better performance in identifying malignant breast tumors, minimizing the risk of false negatives. The selection of the final model can be based on the specific requirements and priorities of the classification task.

The initial results from the models trained on the imbalanced dataset showed relatively low accuracy and imbalanced F1 scores across the sentiment classes. To address this, undersampling, oversampling, and SMOTE techniques were applied.

After balancing the dataset, the Random Forest Classifier model combined with oversampling achieved the best results, with an impressive accuracy of 94% and high F1 scores for each sentiment class (0: 0.92, 1: 0.93, 2: 0.97). This indicates the effectiveness of oversampling in improving the performance of sentiment analysis.

It is important to note that the choice of the best model and balancing technique may vary depending on the dataset and problem domain. Therefore, thorough evaluation and experimentation are recommended to identify the most suitable approach for sentiment analysis in specific contexts.
## Conclusion
In conclusion, this project demonstrates the application of sentiment analysis techniques to US airline customer tweets and showcases the impact of balancing techniques on the performance of machine learning models. By analyzing customer sentiment, airlines can gain valuable insights into customer experiences and tailor their services accordingly.

Feel free to customize this README content based on your project and add any additional details or sections as needed.