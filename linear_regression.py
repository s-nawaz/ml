# Linear regression from scratch
# Two methods for linear regression:
#1. Uses linear algebra operations to perform fast matrix operations to optimize with
# gradient descent. Mean squared error is the loss function
# Loss is minimized with incremental improvement in weights W at learning rate alpha
#2. Uses linear algebra operations to perform fast matrix operations to solve simultaneous linear equations.
# Uses normal equation menthod
# Make sure there are only numerical features. Object types features will be deleted.
# Author: Shahnawaz Alam

import pandas as pd
import numpy as np
import sys, math
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def norma( d ):
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    d_ = d.drop( target_column )
    rescaledx = minmax_scaler.fit_transform( d_ )
    rd = pd.DataFrame( rescaledx, columns=d_.columns )

def get_XY( d, target_column=None ):
    if not target_column or not batch_size:
        target_column = d.columns[-1]
    d_ = d.drop( target_column, axis=1 )
    train_X = d_.to_numpy()
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    train_X = minmax_scaler.fit_transform( train_X )
    bias_ones = np.ones( ( train_X.shape[0], 1 ) )
    print( np.isnan( train_X ).sum().sum() )
    train_X = np.concatenate( ( bias_ones, train_X ), axis=1 )
    print( 'train_X: ', train_X[:10] )
    train_Y = d[ target_column ].to_numpy()
    print( 'train_Y: ', train_Y[:10] )
    return train_X, train_Y

def regress_normal_eq( d, target_column=None ):
    train_X, train_Y = get_XY( d )
    inv_ = np.linalg.inv( np.dot( train_X.transpose(), train_X) )
    print( inv_.shape )
    w = np.dot( inv_, np.dot( train_X.transpose(), train_Y ) )
    w = np.expand_dims( w, axis=1 )
    print( w.shape )
    return w


def lregress( d, alpha=.2, epochs=1000, batch_size=None, target_column=None ):
    if not batch_size:
        batch_size = d.shape[0]
    train_X, train_Y = get_XY( d, target_column )
    W = 1/100. * np.random.rand( 1, train_X.shape[1] )
    print( 'X:', train_X.shape )
    print( 'Y:', train_Y.shape )
    print( 'W:', W.shape )
    print( 'W:', W )
    for i in np.arange( 1, epochs+1 ):
        y_hat = ( W * train_X ).sum( axis=1 )
        #print( 'train_Y: ', train_Y[:10] )
        #print( 'y_hat: ', y_hat.shape )
        #print( 'y_hat: ', y_hat[:10] )
        y_diff = ( y_hat - train_Y ).reshape( train_Y.shape[0], 1 )
        #print( 'y_diff:', y_diff.shape )
        #print( 'y_diff: ', y_diff[:10] )
        loss = 1 / y_diff.shape[0] * ( y_diff * y_diff ).sum()
        if i % 100 == 0:
            print( 'Epoch: ', i, 'loss: ', loss )
        # calculate gradient
        J = ( y_diff * train_X )
        #print( 'J: ', J.shape )
        #print( 'J: ', J[:10] )
        J = J.sum( axis=0 ).reshape( 1, train_X.shape[1] )
        #print( 'J: ', J )
        J = 1 / y_diff.shape[0] * J
        #print( 'J: ', J )
        W = W - alpha * J
        #print( 'W: ', W )
    W = W.transpose()
    return W

def predict( w, td ):
    # Remove the target class. Only required if the target class is present in the test file.
    # We are doing this only for testing the train file.
    test_X, test_Y = get_XY( td )
    predicted = np.dot( test_X, w )
    return predicted

if __name__=='__main__':
    if len( sys.argv ) < 2:
        print( 'Usage:' )
        print( 'For training only: python linear_regression.py <SG/normal> <train_data_file.csv> <learning_rate> <epochs>' )
        print( 'For training and prediction: python linear_regression.py <SG/normal> <train_data_file.csv>  <learning_rate> <epochs> <test_file.csv>' )
    else:
        train_data_file_name = sys.argv[2]
        d = pd.read_csv( train_data_file_name )
        # get rid of object type columns
        d = d.select_dtypes( exclude=[ 'object' ] )
        d.fillna( 0, inplace=True )
        target_column = d.columns[-1]
        print( d.columns )
        print( d.describe() )
        print( d.info() )
        test_file = None
        if sys.argv[1] == 'GD':
            alpha = float( sys.argv[3] )
            epochs = int( sys.argv[4] )
            if len( sys.argv ) == 6:
                test_file = sys.argv[5]
            w = lregress( d, alpha, epochs )
        elif sys.argv[1] == 'normal':
            if len( sys.argv ) == 4:
                test_file = sys.argv[3]
            w = regress_normal_eq( d )
        else:
            print( 'Unknown method.' )
        if test_file:
            print( 'Predicting ...' )
            td = pd.read_csv( test_file )
            # get rid of object type columns
            td = td.select_dtypes( exclude=[ 'object' ] )
            td.fillna( 0, inplace=True )
            predicted_values = predict( w, td )
            td[ 'predicted_value' ] = predicted_values
            td.to_csv( 'lr_predicted.csv', index=False )
            print( 'Output written to lr_predicted.csv' )
