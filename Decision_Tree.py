from Node import *
import queue

class Decision_Tree_Classifier:
    
    max_depth = 0
    info_gain_threshold = 0
    node_size_threshold = 0
    node_entropy_threshold = 1
    head = None
    
    
    def __init__(self, depth, info_gain, node_size, train_df, feature_dict):
        self.max_depth = depth
        self.info_gain_threshold = info_gain
        self.node_size_threshold = node_size
        self.head = Binary_Classifier_Node(train_df, feature_dict, 0)
        self.feature_types = dict(feature_dict)
        
    def create_children(self, node):
        if (node.size < self.node_size_threshold or node.depth >= self.max_depth or node.get_entropy(node.data_pts) == 1):
            return
        children = node.choose_split()
        if (children[3] < self.info_gain_threshold):
            return
        split_feature = children[1]
        split_children = children[4]
        
        for child in split_children.keys():
            
            # make a deep copy for data protection for children
            child_features = dict(node.feature_dict)
            child_features.pop(split_feature)
            child_node = Binary_Classifier_Node(split_children[child], child_features, node.depth + 1)
            node.children.append(child_node)
            
    def construct_tree(self):
        node_queue = queue.Queue()
        
        node_queue.put(self.head)
        while (not node_queue.empty()):
            node = node_queue.get()
            self.create_children(node)
            for child in node.children:
                node_queue.put(child)
    
    def predict(self, input_features):
        temp_node = self.head
        while (len(temp_node.children) != 0):
            temp_node = temp_node.choose_child(input_features[temp_node.feature])
        return temp_node.data_pts['label'].mode().iloc[0]
    
class Decision_Tree_Regressor:
    
    max_depth = 0
    info_gain_threshold = 0
    node_size_threshold = 0
    node_entropy_threshold = 1
    head = None
    
    
    def __init__(self, depth, info_gain, node_size, train_df, feature_dict):
        self.max_depth = depth
        self.info_gain_threshold = info_gain
        self.node_size_threshold = node_size
        self.head = Regression_Node(train_df, feature_dict, 0)
        self.available_features = set(feature_dict.keys())
        
    def create_children(self, node):
        if (node.size < self.node_size_threshold or node.depth >= self.max_depth or node.get_entropy(node.data_pts) == 1):
            return
        children = node.choose_split()
        if (children[3] < self.info_gain_threshold):
            return
        split_feature = children[1]
        split_children = children[4]
        for child in split_children.keys():            
            # make a deep copy for data protection for children
            child_features = dict(node.feature_types)
            child_features.pop(split_feature)
            child_node = Regression_Node(split_children[child], child_features, node.depth + 1)
            node.children[child] = child_node
            
    def construct_tree(self):
        node_queue = queue.Queue()
        
        node_queue.put(self.head)
        while (not node_queue.empty()):
            node = node_queue.get()
            self.create_children(node)
            for child in node.children.keys():
                node_queue.put(node.children[child])
    
    def predict(self, input_features):
        temp_node = self.head
        while (len(temp_node.children) != 0):
            temp_node = temp_node.choose_child(input_features[temp_node.feature])
        return temp_node.data_pts['label'].mean()
                
        