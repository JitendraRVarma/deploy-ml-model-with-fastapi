# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* Model Name: SalaryClassifier
* Model Type: Random Forest Classifier
* Model Version: 1.0
* Model Author: Jitendra Varma
* Date Created: 22-Aug-23
* Last Updated: 22-Aug-23

## Intended Use
The SalaryClassifier is designed to classify instances from the provided dataset into two categories: '>50K' and '<=50K'. It's intended to assist in predicting income levels based on various features such as age, education, workclass, etc.

## Training Data
* Data Source: https://archive.ics.uci.edu/dataset/20/census+income
* Data Size: 48842
* Data Collection Process: Extraction was done by Barry Becker from the 1994 Census database.  A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

## Evaluation Data
* Data Source: https://archive.ics.uci.edu/dataset/20/census+income
* Data Size: 48842
* Data Collection Process: Extraction was done by Barry Becker from the 1994 Census database.  A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

## Metrics
ROC-AUC: Receiver Operating Characteristic Area Under the Curve, measuring the model's ability to distinguish between positive and negative classes.

The model's performance is evaluated using the following metrics:

ROC-AUC (3-fold Cross-Validation):
Fold 1: 0.7697
Fold 2: 0.7797
Fold 3: 0.7792
### Performance

Average ROC-AUC: 0.7762
This section would provide details about the ROC-AUC scores obtained through 3-fold cross-validation, broken down by each fold. The average ROC-AUC score across the folds is also provided for a summary measure of the model's performance.
## Ethical Considerations
* Potential Biases: The model's predictions may be influenced by biases present in the training data, which could result in disparities among different groups.
* Fairness Concerns: Care should be taken to ensure that the model's predictions do not disproportionately affect certain demographic groups.
* Privacy Concerns: While the model doesn't explicitly handle sensitive data, the input features might indirectly reveal personal information.
* Accountability: Model predictions are solely based on the provided data and might not be suitable for making high-stakes decisions without human review.

## Caveats and Recommendations
* Model Limitations: The model's predictions heavily rely on the quality and representativeness of the training data. It might not generalize well to unseen or highly variable data.
* Regular Monitoring: Continuously monitor the model's performance and update it as needed to maintain accuracy and fairness.
* External Factors: Keep in mind that the model's predictions can be affected by external factors that it's not aware of.