Decision Tree

Author: Shahnawaz Alam

Takes a training dataset, builds a decision tree and predicts the class of test data using the learned tree.
Training data file should be comma separated values. First column should be row id and last
column should be class.
Test file should be like the training file. class column is not required in this file.
Information gain maetric is also claculated for comparison.

python> import decision_tree as dt

tree = dt.train( <training data file> )
  
tree.print_tree()

d = dt.predict( tree, <test data file> )
  
