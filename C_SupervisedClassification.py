import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.model_selection import train_test_split


"""""
This file contains two functions: logregression_combined, and randomforest_fun
"""""

"""""
`logregression_combined` function: 
This function runs multinomial logistic regression analysis on a dataset with both numerical and 
categorical variables. It employs multinomial logistic regression to process each variable as follows:
    - Numerical Variables: The function performs multinomial logistic regression using each numerical variable as 
        a single predictor against a multi-level outcome variable (the pseudo-label obtained from clustering).
    - Categorical Variables: For categorical variables, the function converts each category into dummy variables. 
        These dummy variables are then used in the multinomial logistic regression. Please note that more levels in a 
        categorical variable will result in a larger number of dummy variables, which automatically gives a higher F1 score. 
        Therefore, this function returns, next to the F1 score also the Akaike Information Criterion (AIC),
        and the Bayesian Information Criterion (BIC) to assess model performance and complexity.

Parameters:
    - `dataset`: DataFrame containing the data. This dataset should have the numerical variables standardized.
        The categorical variables should NOT be dummy coded.
    - `label`: The pseudo-labels obtained from clustering. The labels should be of class categorical.
    - `sort_by` (optional): Criterion to sort the results by ('AIC', 'BIC', or 'f1-score'), default is 'BIC'.

Returns:
    - A DataFrame sorted by the specified criterion that lists each variable along with its corresponding F1 score, 
        AIC, and BIC.

This function is particularly useful for identifying the most influential predictors in your dataset and 
for balancing model complexity with predictive performance.
"""""


def logregression(dataset, label, sort_by ='BIC', seed = 111): #sort_by options: AIC, BIC, or f1-score
    results = {
        'Variable': [],
        'f1-score': [],
        'AIC': [],
        'BIC': []
    }

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state= seed)

    # Iterate through each variable in the dataset
    for column in dataset.columns:
        # Prepare data for regression
        if dataset[column].dtype == 'object' or dataset[column].dtype == 'category':
            # Handle categorical data by creating dummy variables
            X_train_column = pd.get_dummies(X_train[column], prefix=column)
            X_test_column = pd.get_dummies(X_test[column], prefix=column)
        else:
            # Handle numerical data by selecting the column as is
            X_train_column = X_train[[column]]
            X_test_column = X_test[[column]]

        # Fit a logistic regression model
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        model.fit(X_train_column, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test_column)
        y_pred_proba = model.predict_proba(X_test_column)

        # Calculate the negative log likelihood
        nll = log_loss(y_test, y_pred_proba, normalize=False)

        # Calculate number of parameters (coefficients plus intercept)
        num_params = model.coef_.shape[0] * model.coef_.shape[1] + model.intercept_.shape[0]

        # Calculate AIC and BIC
        aic = 2 * num_params + nll
        bic = np.log(X_test_column.shape[0]) * num_params + nll

        # Get classification report in dictionary format
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Store the results
        results['Variable'].append(column)
        results['f1-score'].append(report_dict['weighted avg']['f1-score'])
        results['AIC'].append(aic)
        results['BIC'].append(bic)

    # Convert the results dictionary to a DataFrame for a nice table format
    results_df = pd.DataFrame(results)

    # Sort the DataFrame in ascending or descending order (depending on if F1 score or BIC/AIC is chosen)
    answer = not sort_by == 'f1-score'
    sorted_results_df = results_df.sort_values(by=f'{sort_by}', ascending=answer)

    return sorted_results_df

"""
The `randomforest_fun` function applies a Random Forest classifier to a standardized dataset to predict labels and 
evaluates the importance of each feature within the model.

Procedure:
    - Model Fitting: The function fits a Random Forest model to the entire dataset using default settings, 
        with a specified random state for reproducibility.
    - Feature Importance Calculation: After fitting, it calculates the importance of each feature in the dataset. 
        Feature importance is derived from how much each feature contributes to decreasing the weighted impurity in 
        splits during the construction of the trees in the model.
    - Visualization: It then visualizes these importances using a horizontal bar plot, which helps in easily 
        identifying which features have the most significant impact on the prediction.
    - Data Output: The function returns a sorted list of feature importances and a dictionary that maps each feature 
        to its importance, sorted in descending order. This allows for easy access to the features that contribute most 
        to the model.

Parameters:
    - `standardized_dataset`: DataFrame. Numerical variables should be standardized. Categorical variables should be
        one-hot encoded.
    - `labels`: Pseudo-labels from clustering. The labels should be of class integer.

Returns:
    - A tuple containing two elements:
      1. A sorted list of tuples, each containing a feature name and its importance.
      2. A dictionary where each feature name is keyed to its importance, sorted in descending order.

This function is useful for understanding the contribution of each feature to the model predictions, 
particularly in the context of feature selection or model interpretation.
"""

def randomforest_fun(standardized_dataset, labels, seed = 111, max_visualized = 100, label_size = 6,
                     title = 'Feature Importances for Cluster Labels'):
    X = standardized_dataset
    y = labels

    # Fit a random forest model
    model = RandomForestClassifier(random_state=seed)
    model.fit(X, y)

    # Get feature importances
    importances = model.feature_importances_

    # Ensure the correct feature names are used
    feature_names = standardized_dataset.columns.tolist()

    # Create a dictionary of feature names and their importances
    feature_importance_dict = dict(zip(feature_names, importances))

    # Sort the dictionary by its values in descending order
    sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)

    # Convert the sorted items back into a dictionary
    sorted_feature_importance_dict = dict(sorted_feature_importance)

    # Splitting the keys and values for plotting
    sorted_features = [item[0] for item in sorted_feature_importance]
    sorted_importances = [item[1] for item in sorted_feature_importance]

    # Plot feature importances in descending order
    plt.figure(figsize=(8, 14))
    sns.barplot(x=sorted_importances[0:max_visualized], y=sorted_features[0:max_visualized])
    plt.title(f'{title}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tick_params(axis='y', labelsize=label_size)  # Set y-axis label size to smaller font
    plt.show()

    return sorted_feature_importance, sorted_feature_importance_dict
