# Introduction 

In deepchecks there are multiple test suites already available which can check the train vs test model metrics , datasets characteristics and  much more. One example of such suite is DatasetsSizeComparison.

DatasetsSizeComparison
Conditions:
0: Test-Train size ratio is greater than 0.01

These suites contains conditions which can also be edited , changed or new added. But this preavialable suite contains codition which checks that test data and train data ratio is greater than 0.01.


## Types
So there are primarily three types of checks taking place in Deepcheck in the process. They are as follows :
1) Data Integrity Check 
2) Check for distribution of data for train and test
3) Model Performance Evaluation for unseen data or close to real-world data

Dataset_integrity checks the integrity present in the dataset such as missing fields and etc.
Train Test validation set checks the correctness of the split of the data for training and testing phase.
Lastly model evaluation cross-checks the model performance and genericness and also the signs of overfitting if present.
