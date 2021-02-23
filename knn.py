# k Nearest Neighbors classification
# Author: Shahnawaz Alam
# Predicts the class of test data based on majority class of k nearest neighbors
# Based on matrix operations. This eliminates loops and speeds up the algorithm.
# For usage: python.py knn.py

import pandas as pd
import numpy as np
import sys

def get_dataset( f_name ):
    d = pd.read_csv( f_name, header=0 )
    return d

def euclidean_distance( train_data, test_data ):
    train_minus_test = train_data - test_data
    train_minus_test_squared = train_minus_test * train_minus_test
    euclid_dist = np.sqrt( np.sum( train_minus_test_squared, axis=2 ) )
    return euclid_dist

def knn( k, train_file, test_file ):
    d = get_dataset( train_file )
    class_column = d.columns[ -1 ]
    # Pick up only the feature columns
    training_data = d[ d.columns[ 1:5 ] ].to_numpy()
    test = get_dataset( test_file )
    # Make sure that the order of feature columns is the same as the training data
    test_data = test[ test.columns[ 1:5 ] ].to_numpy()
    test_data = np.expand_dims( test_data, axis=1 )
    euclid_dist = euclidean_distance( training_data, test_data )
    # Add the distance columns to the training data set for viewing
    #for i in range( euclid_dist.shape[0] ):
    #    d[ 'd'+str(i) ] = euclid_dist[ i ]
    #print( d )
    # get the args of the smallest distances
    ksmallest = np.argpartition( euclid_dist, k )[ :,:k ]
    # Print the vectors in training data set that nearest to the test data vector
    #for i in range( ksmallest.shape[0] ):
    #    print( d.iloc[ ksmallest[i] ] )
    # Get the class of the majority of the nearest ones
    pred_list = list( map( lambda t: d.iloc[ t ][ class_column ].mode()[0], ksmallest ) )
    test[ 'predicted_class' ] = pred_list
    #print( test )
    return test

if __name__=='__main__':
    if len( sys.argv ) < 4:
        print( 'Usage: python knn.py <k> <trainfile.csv> <testfile.csv>' )
        print( 'Training data file should be comma separated values. First column should be row id and last' )
        print( 'column should be class' )
        print( 'Test file should be like the training file. class column is not required in this file' )
    else:
        k = int( sys.argv[1] )
        test = knn( k, sys.argv[2], sys.argv[3] )
        test.to_csv( 'predicted.csv', index=False )
        print( 'Output is in file predicted.csv that has predictions in column predicted_class' )
