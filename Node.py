import pandas as pd
import math
import numpy as np

class Node:
    size = 0
    children = dict()
    data_pts = pd.DataFrame()
    entropy = 0
    # specify whether a feature is continuous or discrete
    feature_types = dict()
    # these are updated later when determining split - used for prediction
    feature_type = None
    split_val = 0
    split_feature = None  
    depth = 0  
    
    def __init__(self, data_pts, feature_dict, depth):
        self.data_pts = pd.DataFrame(data_pts)
        self.size = len(data_pts)
        self.feature_types = dict(feature_dict)
        self.depth = depth
    
    def get_entropy(self, data_pts):
        return NotImplementedError("Implemented in children")
        
    def get_info_gain(self, children):
        parent_entropy = self.get_entropy(self.data_pts)
        for child in children.keys():
            weighted_purity = (len(children[child]) / len(self.data_pts)) * (self.get_entropy(children[child]))
            parent_entropy -= weighted_purity
        return parent_entropy
        
    def choose_split(self):
        feature_columns = self.data_pts.drop('label', axis=1).columns
        best_split_feature = feature_columns[0]
        split_children = dict()
        max_info_gain = 0
        cont_split_val = 0
        split_type = None
        for feature in feature_columns:
            # if the feature is discrete use discrete split technique
            if (self.feature_types[feature] == "D"):
                children = dict()
                unique_vals = self.data_pts[feature].unique()
                info_gain = 0
                for val in unique_vals:
                    temp_df = self.data_pts[self.data_pts[feature] == val]
                    temp_df = temp_df.drop(feature, axis=1)
                    children[val] = temp_df
                info_gain = self.get_info_gain(children)
                if (info_gain > max_info_gain):
                    max_info_gain = info_gain
                    best_split_feature = feature
                    split_children = children
                    split_type = "D"
            # otherwise it's continuous - use continuous split technique
            else:
                children = dict()
                unique_vals = self.data_pts[feature].unique()
                unique_vals = np.sort(unique_vals)
                index = 0
                for i in unique_vals[1:]:
                    split = (i + unique_vals[index]) / 2
                    child_1 = self.data_pts[self.data_pts[feature] > split]
                    child_2 = self.data_pts[self.data_pts[feature] <= split]
                    child_1 = child_1.drop(feature, axis=1)
                    child_2 = child_2.drop(feature, axis=1)
                    children[0] = child_1
                    children[1] = child_2
                    info_gain = self.get_info_gain(children[val])
                    if (info_gain > max_info_gain):
                        max_info_gain = info_gain
                        best_split_feature = feature
                        split_children = children
                        cont_split_val = split
                        split_type = "C"
                    index += 1
                
        self.split_feature = best_split_feature
        self.split_val = cont_split_val
        self.feature_type = split_type
        return (split_type, best_split_feature, cont_split_val, max_info_gain, split_children)
    
    def choose_child(self, feature_val):
        if (self.feature_type == 'D'):
            return self.children[feature_val]
        else:
            if (feature_val < self.split_val):
                return self.children[0]
            else:
                return self.children[1]
            
    
class Binary_Classifier_Node(Node):
    def get_entropy(self, data_pts):
            if (len(data_pts) == 0):
                return 0
            prob = (len(data_pts[data_pts['label'] == 1])) / len(data_pts)
            if (prob == 1 or prob == 0):
                return 0
            else:
                return (prob * math.log2(prob)) - ((1 - prob) * math.log2(1 - prob))

class Regression_Node(Node):
    def get_entropy(self, data_pts):
        return data_pts['label'].var()
        
        
    
    
    