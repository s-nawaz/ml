# Decision Tree
# Author: Shahnawaz Alam
# Takes a training dataset, builds a decision tree and predicts the class of test data using the learned tree.
# Training data file should be comma separated values. First column should be row id and last
# column should be class
# Test file should be like the training file. class column is not required in this file
# Information gain maetric is also claculated for comparison


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
import math, itertools, sys

MIN = int( -10e10 )
MAX = int( 10e10 )

class Node:
    node_id = itertools.count()
    def __init__( self, isleaf=False, feature_name=None, metrics=None, feature_names_remaining=[], feature_type=None, feature_value_type=None, split_data=None, next_nodes=None, class_column=None, decision_class=None ):
        # If isleaf is True, the node is a leaf node. A decision class has to be assigned to this node.
        # The class with highest occurrence in the dataset is the decision class.
        self.isleaf = isleaf
        if isleaf:
            self.class_column = class_column
            # Find the class with maximum occurrence in the dataset
            classes = split_data[ class_column ].value_counts()
            self.decision_class = classes.index[ classes.argmax() ]
        else:
            self.feature_name = feature_name
            self.feature_names_remaining = feature_names_remaining
            self.feature_type = feature_type
            self.metrics = metrics
            if next_nodes is None:
                if feature_type == 'discrete':
                    next_nodes = pd.DataFrame( columns={ 'id', 'feature_value', 'next', 'next_node_id' } ) 
                    next_nodes = next_nodes.astype( { 'id': 'int32', 'feature_value': feature_value_type , 'next_node_id': 'int32' } ) 
                    for t in split_data:
                        next_nodes = next_nodes.append( { 'id': t[ 'id' ], 'feature_value': t[ 'feature_value' ] }, ignore_index=True )
                if feature_type == 'continuous':
                    next_nodes = pd.DataFrame( columns={ 'id', 'lower', 'upper', 'next', 'next_node_id' } ) 
                    next_nodes = next_nodes.astype( { 'id': 'int32', 'lower': feature_value_type, 'upper': feature_value_typei, 'next_node_id': 'int32' } ) 
                    for t in split_data:
                        next_nodes = next_nodes.append( { 'id': t[ 'id' ], 'lower': t[ 'interval' ][ 'lower' ], 'upper': t[ 'interval' ][ 'upper' ] }, ignore_index=True )
            self.next_nodes = next_nodes
            self.split_data = split_data
        self.id = next( Node.node_id )

    def next( self, feature_val ):
        if self.isleaf:
            return None
        next_node = None
        next_node_ = None
        if self.feature_type == 'discrete':
            next_node_ = self.next_nodes.loc[ self.next_nodes[ 'feature_value' ] == feature_val ]
        elif self.feature_type == 'continuous':
            next_node_ = self.next_nodes.loc[ ( feature_val > self.next_nodes[ 'lower' ] ) & ( feature_val <= self.next_nodes[ 'upper' ] ) ] 
            if  next_node_.empty:
                if feature_val <= MIN:
                    next_node_ = self.next_nodes.iloc[ [0] ]
                elif feature_val > MAX:
                    next_node_ = self.next_nodes.iloc[ [-1] ]
        if next_node_ is not None and next_node_ is not np.nan and not next_node_.empty:
            next_node = next_node_.iloc[ 0 ][ 'next' ]
        return next_node

    def add_next_node( self, i, next_node ):
        self.next_nodes.loc[ self.next_nodes[ 'id' ] == i, 'next' ] = next_node
        self.next_nodes.loc[ self.next_nodes[ 'id' ] == i, 'next_node_id' ] = int( next_node.id )

    def get_next_nodes( self ):
        nodes = []
        if self.next_nodes is not None:
            for idx, row in self.next_nodes.iterrows():
                nodes.append( row[ 'next' ] )
        return nodes

    def set_next( self, next_nodes ):
        self.next_nodes

    def print_node( self ):
        print( 'Node: ', self.id, ' ', 'isleaf: ', self.isleaf, ' ', self, '--------------' )
        if not self.isleaf:
            print( 'Feature name: ', self.feature_name, ' Feature type: ', self.feature_type )
            print( 'metrics: ', self.metrics )
            print( 'Features remaining: ', self.feature_names_remaining )
            print( 'next_nodes: ' )
            print( self.next_nodes.head( 10 ) )
        else:
            print( 'class: ', self.decision_class )
        print( '--------------------------------------------' )

    def print_tree( self ):
        self.print_node()
        #print( type( self.next_nodes ) )
        if not self.isleaf:
            for idx, row in self.next_nodes.iterrows():
                #print( type( row ), row )
                node = row[ 'next' ]
                print( 'node: ', node )
                if node is not np.nan:
                    node.print_tree()

def get_dataset( f_name ):
    d = pd.read_csv( f_name, header=0 )
    return d

# Calculate classification error of a given dataset D
def classification_error( D, class_column='Class' ):
    # Get the class with maximum occurrence
    class_counts = D[ class_column ].value_counts()
    class_with_max_instances = class_counts.index[ class_counts.argmax() ]
    CE = ( D.shape[0] - ( D[ class_column ] == class_with_max_instances ).sum() )
    return CE

# Calculate entropy of a given dataset D
def entropy( D, class_column='Class' ):
    E = 0
    # Get the unique number of classes
    classes = D[ class_column ].unique()
    # If there is only one class, entropy is 0
    if len( classes ) == 1:
        return E
    for c in classes:
        # Calculate the probability of class c
        d_ = D[ D[ class_column ]==c ]
        proba = d_.shape[0] / D.shape[0]
        plogp = -1 * proba * math.log( proba, 2 )
        E += plogp
    return E

# Calculate Information Gain
# For continuous value feature, we take the median as
# the split point
def IGain( D, feature_name, class_column ):
    # Wighted average of entropy for each interval
    wa_entropy = 0
    total_CE = 0
    split_data = []
    feature_type = None
    feature_value_type = D[ feature_name ].dtype.name
    if feature_value_type.startswith( 'float' ) or feature_value_type.startswith( 'int' ):
        feature_type = 'continuous'
    else:
        feature_type = 'discrete'
    entropyD = entropy( D, class_column )
    if feature_type == 'continuous':
        median = D[ feature_name ].median()
        # There are two categories of the feature: <= median and > median
        split_intervals = [ MIN ]
        split_points = [ median ]
        split_intervals += split_points + [ MAX ]
        for i in range( len( split_intervals ) - 1 ):
            lower, upper = split_intervals[ i ], split_intervals[ i + 1 ]
            # Split the data
            D_ = D[ D[ feature_name ] > lower ]
            D_ = D_[ D_[ feature_name ] <= upper ]
            entropyD_ = entropy( D_, class_column )
            CE = classification_error( D_, class_column )
            split_data.append( { 'id': i, 'dataset': D_, 'classification_error': CE, 'entropy': entropyD_, 'interval': { 'lower': lower, 'upper': upper } } )
            wa_entropy += ( D_.shape[0] / D.shape[0] ) * entropyD_
            total_CE += CE
    else:
        # Find the unique value of the feature
        unique_values = list( D[ feature_name ].unique() )
        for i, uv in enumerate( unique_values ):
            # Split the data
            D_ = D[ D[ feature_name ] == uv ]
            entropyD_ = entropy( D_, class_column )
            CE = classification_error( D_, class_column )
            split_data.append( { 'id': i, 'dataset': D_, 'classification_error': CE, 'entropy': entropyD_, 'feature_value': uv } )
            #input()
            wa_entropy += ( D_.shape[0] / D.shape[0] ) * entropyD_
            total_CE += CE
        
    igain = entropyD - wa_entropy
    total_CE = total_CE / D.shape[0]
    return igain, total_CE, feature_type, feature_value_type, split_data

def create_node( d, feature_names, class_column ):
    node = None
    split_data = []
    requires_splitting = len( d[ class_column ].unique() ) > 1
    # If no more feature is available and no splitting is required, create a leaf node and assign a class to the leaf.
    # The class with highest occurrence in the dataset is the chosen for this leaf.
    if not( len( feature_names ) and requires_splitting ):
        # Create leaf node
        node = Node( isleaf= True, split_data=d, class_column=class_column )
    # If there is a feature based on which no decision has yet been added to the tree,
    # then select the feature that provides the maximum information gain and split the data
    # based on that feature.
    else:
        metrics = pd.DataFrame( columns=[ 'feature_name', 'information_gain', 'classification_error' ] )
        split_data_list = []
        metrics[ 'feature_name' ] = feature_names
        # Calculate information gain for each feature
        for feature_name in feature_names:
            information_gain, classification_error, feature_type, feature_value_type, split_data = IGain( d, feature_name, class_column )
            metrics.loc[ metrics[ 'feature_name'] == feature_name, 'information_gain' ] = information_gain
            metrics.loc[ metrics[ 'feature_name'] == feature_name, 'classification_error' ] = classification_error
            split_data_list.append( { 'feature_name': feature_name, 'feature_type': feature_type, 'feature_value_type': feature_value_type, 'split_data': split_data } )
            #information_gains.loc[ information_gains[ 'feature_name'] == feature_name, 'split_data' ] = split_data
        metrics = metrics.astype( { 'information_gain': 'float32', 'classification_error': 'float32' } )
        # Find the feature witht the largest information gain
        #idx = metrics[ 'information_gain'].argmax()
        # Find the feature witht the smallest classification error
        idx = metrics[ 'classification_error'].argmin()
        # Found the feature with the maximum information gain: id_max
        #split_data = information_gains.iloc[ id_max ][ 'split_data' ]
        t = next( ( t for t in split_data_list if t[ 'feature_name' ] == metrics.iloc[ idx ][ 'feature_name' ] ), None )
        split_data = t[ 'split_data' ]
        # Construct the node with the given feature with the ranges of split feature values
        #feature_name = information_gains.iloc[ id_max ][ 'feature_name' ]
        feature_name = t[ 'feature_name' ]
        feature_type = t[ 'feature_type' ]
        feature_value_type = t[ 'feature_value_type' ]
        #feature_type = information_gains.iloc[ id_max ][ 'feature_type' ]
        #feature_value_type = information_gains.iloc[ id_max ][ 'feature_value_type' ]
        feature_names_remaining = [ t for t in feature_names if t not in [ feature_name ] ]
        classification_error = metrics.iloc[ idx ][ 'classification_error' ]
        # Create branch node
        node = Node( feature_name=feature_name, metrics=metrics, feature_names_remaining=feature_names_remaining, feature_type=feature_type, feature_value_type=feature_value_type, split_data=split_data )
        # next_nodes of the node will be populated when they are constructed
    return node

def build_tree( node, class_column ):
    for d_ in node.split_data:
        node_ = create_node( d_[ 'dataset' ], node.feature_names_remaining, class_column )
        if node_ is not None:
            node.add_next_node( d_[ 'id' ], node_ )
    # Build tree recursively
    for node_ in node.get_next_nodes():
        if node_ is not np.nan and not node_.isleaf:
            build_tree( node_, class_column )
    return

def train( train_file ):
    d = get_dataset( train_file )
    num_columns = len( d.columns )
    feature_names = list( d.columns[ 1:num_columns-1 ] )
    class_column = d.columns[ num_columns-1 ]
    Tree = create_node( d, feature_names, class_column )
    if Tree is not None and Tree is not np.nan and not Tree.isleaf:
        build_tree( Tree, class_column )
    return Tree

def predict( Tree, test_file ):
    d = get_dataset( test_file )
    d[ 'predicted_class' ] = ''
    for idx, row in d.iterrows():
        # Traverse the Tree to reach a final node and predict the class
        node = Tree
        while not node.isleaf:
            #node.print_node()
            node = node.next( row[ node.feature_name ] )
        if node.isleaf:
            decision_class = node.decision_class
            d.loc[ idx, 'predicted_class' ] = decision_class
        else:
            print( 'Could not find a leaf node ...' )
    return d

if __name__=='__main__':
    if len( sys.argv ) < 2:
        print( 'Usage for training only : python id3.py <trainfile.csv>' )
        print( 'Usage for training and prediction : python id3.py <trainfile.csv> <testdatafile.csv' )
        print( 'Training data file should be comma separated values. First column should be row id and last' )
        print( 'column should be class' )
        print( 'Test file should be like the training file. class column is not required in this file' )
    else:
        Tree = train( sys.argv[1] )
        print( 'Tree created.' )
        if len( sys.argv ) > 2:
            d = predict( Tree, sys.argv[2] )
            d.to_csv( 'predicted.csv', index=False )
            print( 'Output is in file predicted.csv that has predictions in column predicted_class' )
