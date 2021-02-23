Author: Shahnawaz Alam

1. Decision Tree

Takes a training dataset, builds a decision tree and predicts the class of test data using the learned tree.
Training data file should be comma separated values. First column should be row id and last
column should be class. Works with both continuous and discrete feature values. In case of continuous values, median is used for splitting.
Test file should be like the training file. Class column is not required in this file.
Data can be split using the following menthods:

(i) Information Gain

(ii) Classification Error

(iii) Gain Ratio

(iv) Gini Index

python> import decision_tree as dt

tree = dt.train( training_data_file, 'IG' ) # for Information Gain

tree = dt.train( training_data_file, 'CE' ) # for Classification Error

tree = dt.train( training_data_file, 'GR' ) # for Gain Ratio

tree = dt.train( training_data_file, 'Gini' ) # for Gini Index

tree.print_tree()

d = dt.predict( tree, test_data_file )


2. k Nearest Neighbors (kNN)

Predicts the class of test data based on majority class of k nearest neighbors.

Based on matrix operations. This eliminates loops and speeds up the algorithm.

python> import knn

result = knn.knn( k, training_data_file, test_data_file )
