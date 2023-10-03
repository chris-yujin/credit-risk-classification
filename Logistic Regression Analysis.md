# Module 12 Logistic Regression Report

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
The purpose of this analysis is to identify credit worthiness of borrowers.

* Explain what financial information the data was on, and what you needed to predict.
The financial information includes: loan size, interest rate, borrower income, debt to income ratio, number of accounts, derogatory marks, total debt, and loan status. We are trying to predict the loan status. 

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
For loan status we split the data into two sets (training and testing). According to the value counts we have 75036 training targets and 2500 testing targets.

* Describe the stages of the machine learning process you went through as part of this analysis.
The general process of the machine learning process is as follow
  * split data into training and testing sets
  * import LogisticRegression from the sklearn library and assign it to a random state for repeatablitly.
  * Using the X_train and y_train data fit your model 
  * Predict the outcomes using the X_test data only and save the predictions
  * Evaluate the performance of the model by comparing the predictions using the X_test data to the predictions from the training dataset (y_train)
  * Reflect and tune appropriately.

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).


The fit method is a fundamental method in scikit-learn that is used to train a machine learning model on a given dataset. It takes two main arguments: the input features (X) and the target labels (y). When you call fit, the model learns the patterns and relationships in the data, adjusting its internal parameters to make predictions.
predict:

The predict method is used to make predictions with a trained machine learning model. Once a model is trained using the fit method, you can use the predict method to provide new input data (features) and get predictions (labels or values) from the model. For example, in a classification task, predict will return the class labels the model assigns to the input data.
Logistic:

In scikit-learn, LogisticRegression is a class that implements logistic regression, which is a common algorithm for binary and multiclass classification problems. You can create an instance of LogisticRegression, then use fit to train it on your data and predict to make predictions. It's part of the linear models module in scikit-learn.
train_test_split:

train_test_split is a function in scikit-learn that is used to split a dataset into two or more subsets for training and testing purposes. It helps in evaluating the performance of a machine learning model. Typically, you provide your dataset along with the desired split ratio, and it randomly divides the data into training and testing sets. This allows you to train the model on one portion of the data and test its performance on another.
RandomOverSampler:

RandomOverSampler is a method used for dealing with class imbalance in classification tasks. Class imbalance occurs when one class in the dataset has significantly fewer examples than another class. This can lead to biased models that perform poorly on the minority class. RandomOverSampler is a technique to balance the class distribution by randomly oversampling the minority class to match the number of samples in the majority class. It can help improve the model's performance on imbalanced datasets.


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
    * Balanced Accuracy Score: 0.94426
    * Accuracy: 0.99
    * Target `0` (healthy loan)
      * Precision: 1.00
      * Recall: 1.00
      * f1-score: 1.00
    * Target `1` (high-risk loan)
      * Precision: 0.87
      * Recall: 0.89
      * f1-score: 0.88

      Confusion Matrix:
      * True Positive: 18679
      * False Positive: 80
      * True Negative: 558
      * False Negative: 67




* Machine Learning Model 2 (resampled):
  * Description of Model 2 Accuracy, Precision, and Recall scores.
    * Balanced Accuracy Score: 0.9959744
    * Accuracy: 1.00
    * Target `0` (healthy loan)
      * Precision: 1.00
      * Recall: 1.00
      * f1-score: 1.00
    * Target `1` (high-risk loan)
      * Precision: 0.87
      * Recall: 1.00
      * f1-score: 0.93

      Confusion Matrix:
      * True Positive: 18668
      * False Positive: 91
      * True Negative: 623
      * False Negative: 2

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?

The second model with the resampled data appears to perform better. It has an overall better balanced accuracy score and significantly reduced the number of false negatives. 

* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

In the case of identiying high risk loans it is much more important to detect individuals that are likely to default on their loan and there for a fals positive has less reprucussions than a false negative as the a false negative may put the lender at risk of a lender not paying back a loan. 

If you do not recommend any of the models, please justify your reasoning.

I would recommend using the second model with the resampled data. 