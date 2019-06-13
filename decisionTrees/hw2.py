import numpy as np
np.random.seed(42)

chi_table = {0.01  : 6.635,
             0.005 : 7.879,
             0.001 : 10.828,
             0.0005 : 12.116,
             0.0001 : 15.140,
             0.00001: 19.511}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.    
    """
    
    # extract the relevent column
    if len(data.shape) == 1:
        label = np.array(data)
    else:
        label = np.array(data[:, -1]) 
        
    number_of_classes = np.array(np.unique(label, return_counts=True)).T # calculate the number of classes [class][number]
    
    sigma = np.square((np.array(number_of_classes[:, -1])) / len(label)) # calculate the sigma according to the formula
    
    gini = 1 - np.sum(sigma)
    
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """
    
    # extract the relevent column
    if len(data.shape) == 1:
        label = np.array(data)
    else:
        label = np.array(data[:, -1]) 
    
    number_of_classes = np.array(np.unique(label, return_counts=True)).T # calculate the number of classes [class][number]
    
    propability = (np.array(number_of_classes[:, -1])) / len(label) # calculate |Si|/|S| according to the formula
    
    entropy = - np.sum(propability.dot(np.log2(propability)))
    
    return entropy

class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.
    
    def __init__(self, feature, value):
        self.feature = feature # column index of criteria being tested
        self.value = value # value necessary to get a true result
        self.children = []
        self.predict = None
        self.instances = {}
        
        
    def add_child(self, node):
        self.children.append(node)
        
    def is_leaf(self):
        return (len(self.children) == 0)        
            
    def prediction(self, data):
        # pure data
        if len(data) == 1:
            self.predict = data[0]
        # stop beacuse of chi value- data isn't pure and therefor it has more rows
        
        else:
            number_of_classes = np.array(np.unique(data, return_counts=True)).T # calculate the number of classes [class][number]
            if len(number_of_classes) == 1:
                self.predict = number_of_classes[0][0]
            else:
                num_of_0 = number_of_classes[0][1]
                num_of_1 = number_of_classes[1][1]
                if num_of_0 > num_of_1:
                    self.predict = number_of_classes[0][0]
                else:
                    self.predict = number_of_classes[1][0]
        
                    
def thresholds(data, feature):
    """
    creates a list of values represents the the average of each consecutive pair of values.
    
    imput:
    - data: the training dataset
    - feature: the index of the current feature
    return:
    - threshold_data: the threshold of the specific feature
    """
    current_feature = np.array(data[:, feature])
    current_feature = np.sort(current_feature)
    threshold_data = [] 
    
    for i in range(len(current_feature) -1):
        current_avg = (current_feature[i] + current_feature[i+1]) / 2
        threshold_data.append(current_avg)
   
    return threshold_data
    
def best_threshold(data, feature, impurity): 
    """
    finds the best threshold, e.g the threshold that gives the best split of data and impurity reduce
    
    imput:
    - data: the training dataset
    - feature: the index of the current feature
    - inpurity: the impurity measure
    
    output:
    - best_goodness: the best impurity reduce found
    - best_thres: the threshold that gave the best impurity reduce
    - best_upper: the list of values bigger or equal to the threshold
    - best_lower: the list of values bigger then the threshold
    """
    
    best_goodness = 0
    best_thres = 0
    best_upper = []
    best_lower = []
    
    current_threshold = thresholds(data, feature)
        
    for i in range(len(current_threshold)):
        upper, lower = split_data(data, feature, current_threshold[i])
        
        impurity_criterion = goodness_of_split(data, upper, lower, impurity)
        
        if impurity_criterion > best_goodness :
            best_goodness = impurity_criterion
            best_thres = current_threshold[i]
            best_upper = upper
            best_lower = lower
            
    return best_goodness, best_thres, best_upper, best_lower
    
    
def split_data(data, feature, threshold):
    """
    splits the data according to the given threshold into 2 tables
    one will be all the values that are larger then the threshold and the other all the values smaller then the hreshold
    
    imput:
    - data: the training dataset
    - feature: the index of the current feature
    - threshold: the current threshold that will split the data
    
    output:
    - upper: the list of values bigger or equal to the threshold
    - lower: the list of values bigger then the threshold 
    """
    
    upper = []
    lower = []
    
    for i in range(len(data)):
        if data[i, feature] < threshold :
            lower.append(data[i])
        else:
            upper.append(data[i])
    
    upper = np.array(upper)
    lower = np.array(lower)
    
    return upper, lower

def chi_square_test(data, data_left, data_right):
    
    """
    calculating chi square value according to the split data
    imput:
    - data: the training dataset
    - data_ledt: the list of values bigger or equal to the threshold
    - data_right: the list of values bigger then the threshold 
    
    output:
    - chi_square value
    """
    num_0_left = 0
    num_1_left = 0
    num_0_right = 0
    num_1_right = 0
    
    key_left, value_left = np.unique(data_left, return_counts=True)
    d_left = dict(zip(key_left, value_left))
    
    key_right, value_right = np.unique(data_right, return_counts=True)
    d_right = dict(zip(key_right, value_right))
    
    if 0 in d_left:
        num_0_left = d_left[0]
    if 1 in d_left:
        num_1_left = d_left[1]
    if 0 in d_right:
        num_0_right = d_right[0]
    if 1 in d_right:
        num_1_right = d_right[1]
        
    pf = np.array([num_0_left, num_0_right]) # number of instances where Xj = f and Y = 0
    nf = np.array([num_1_left, num_1_right]) # number of instances where Xj = f and Y = 1
    
    key, value = np.unique(data, return_counts=True)
    d = dict(zip(key, value))
    
    py0 = d[0] / (len(data))
    py1 = d[1] / (len(data))
    
    df = pf + nf # number of instances where Xj = f
        
    e0 = df * py0
    e1 = df * py1
    
    sigma = ((np.square(pf - e0)) / e0) + ((np.square(nf - e1)) / e1)
   
    return np.sum(sigma)
    

def goodness_of_split(data, upper, lower, impurity):
    """
    calculate the impurity reduce according to the formula and depends on the given impurity measure
    
    imput:
    - data: the training dataset
    - upper: the list of values bigger or equal to the threshold
    - lower: the list of values bigger then the threshold
    - impurity: the impurity measure
    
    output:
    - impurity_criterion according to the impurity measure
    """
    
    upper_size = upper.shape[0]
    lower_size = lower.shape[0]
    data_size = data.shape[0]
    impurity_criterion = impurity(data) - ( ((upper_size / data_size) * impurity(upper)) + ((lower_size / data_size) * impurity(lower)) )
    
    return impurity_criterion
        
def numInstances(data):
    """
    counts the number of ones and zeros in the data
    
    input:
    - data: label column of current data set to be learnd
    
    output:
    - one_num: number of 1 values in the data set
    - zero_num: number of 0 values in the data set
    """
    labels_column = data[:,-1]
    number_of_instances = labels_column.shape[0]
    one_num = np.nonzero(labels_column)[0].shape[0]
    zero_num = number_of_instances - one_num
    return one_num, zero_num


def best_feature(data, impurity):
    """
    finds the best feature that will give us the best impurity reduce

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. 
    
    Output: 
    - feature: the feature that will give us the best split in the tree
    - best_thres: the best threshold
    - left_child: the list that contains the values smaller then the best threshold that was found
    - right_child: the list that contains the values larger then the best threshold that was found
    """
    
    best_goodness = 0
    feature = 0
    best_thres = 0
    left_child = []
    right_child = []
    
    for i in range(data.shape[1] - 1):
        goodness, thres, upper, lower = best_threshold(data, i, impurity)
        if goodness > best_goodness:
            best_goodness = goodness
            feature = i
            best_thres = thres
            left_child = lower
            right_child = upper
     
    return feature, best_thres, left_child, right_child 

def build_tree(data, impurity, chi_value = 1):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    
    if impurity(data) == 0:
        return
    
    feature, best_thres, left_child, right_child = best_feature(data, impurity)
    root = DecisionNode(feature, best_thres)
    
    one_num, zero_num = numInstances(data)
    root.instances[0] = zero_num
    root.instances[1] = one_num
    
    # left tree
    left_feature, left_best_thres, left_child_first, left_child_second = best_feature(left_child, impurity)
    left_node = DecisionNode(left_feature, left_best_thres)
    left_node.prediction(left_child[:,-1])
    
    one_num, zero_num = numInstances(left_child)
    left_node.instances[0] = zero_num
    left_node.instances[1] = one_num
   
    root.add_child(left_node)
    
    # right tree
    right_feature, right_best_thres, right_child_first, right_child_second = best_feature(right_child, impurity)
    right_node = DecisionNode(right_feature, right_best_thres)
    right_node.prediction(right_child[:, -1])
    
    one_num, zero_num = numInstances(right_child)
    right_node.instances[0] = zero_num
    right_node.instances[1] = one_num
    
    root.add_child(right_node)
    
    build_recursion_tree(left_node, left_child, impurity, chi_value)
    build_recursion_tree(right_node, right_child, impurity, chi_value)
    
    return root
    

def build_recursion_tree(father, data, impurity, chi_value):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    
    if impurity(data) == 0: # father is a leaf - pure
        father.prediction(data[:, -1])
        one_num, zero_num = numInstances(data)
        father.instances[0] = zero_num
        father.instances[1] = one_num
        return
    
    
    feature, best_thres, left_child, right_child = best_feature(data, impurity)
    
    if chi_value != 1:
        current_chi_value = chi_square_test(data, left_child[:, -1], right_child[:, -1])
        if current_chi_value <= chi_table[chi_value]:
        
            father.prediction(data[:, -1])
            one_num, zero_num = numInstances(data)
            father.instances[0] = zero_num
            father.instances[1] = one_num
            return
        
    
    # left tree - small values
    left_feature, left_best_thres, left_child_first, left_child_second = best_feature(left_child, impurity)
    left_node = DecisionNode(left_feature, left_best_thres)
    left_node.prediction(left_child[:,-1])
    
    one_num, zero_num = numInstances(left_child)
    left_node.instances[0] = zero_num
    left_node.instances[1] = one_num
  
    father.add_child(left_node)
    
    # right tree - large values
    right_feature, right_best_thres, right_child_first, right_child_second = best_feature(right_child, impurity)
    right_node = DecisionNode(right_feature, right_best_thres)
    right_node.prediction(right_child[:, -1])
    
    one_num, zero_num = numInstances(right_child)
    right_node.instances[0] = zero_num
    right_node.instances[1] = one_num
   
    father.add_child(right_node)
    
    # recursive call for each sub tree
    build_recursion_tree(left_node, left_child, impurity, chi_value)
    build_recursion_tree(right_node, right_child, impurity, chi_value)
    

def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    if node.is_leaf():
        dic_keys = list(node.instances.keys())
        if len(dic_keys) == 1:
            node.predict = dic_keys[0]
        else:
            if node.instances[0] > node.instances[1]:
                node.predict = 0
            else:
                node.predict = 1
        
        return node.predict
    
    while (not node.is_leaf()):
        if instance[node.feature] < node.value:
            node = node.children[0]
        else:
            node = node.children[1]
    
    
    return node.predict
    

def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    
    same_value = 0.0
    number_of_instances = len(dataset)
    
    for instance in dataset:
        
        pred = predict(node, instance)
        if(pred == instance[-1]): 
            same_value += 1

    accuracy = 100  * (same_value / number_of_instances)    

    return accuracy


def list_of_parents(node, current_list, parent):
    #finds all the parents of leafs only
    
    if node.is_leaf():
        current_list.append(parent)
        return
    
    list_of_parents(node.children[0], current_list, node)
    list_of_parents(node.children[1], current_list, node)
    
    return current_list


def number_of_internal_nodes(root):
    # calculate number of internal nodes (without the leafs)
    
    if (root.is_leaf()):
        return 0
    
    return 1 + number_of_internal_nodes(root.children[0]) + number_of_internal_nodes(root.children[1])

           
def post_pruning(root, data_train, data_test):
    """
    calculate the test accuracy of the tree assuming no split occurred on the parent of that leaf and find the best such parent 
    input:
    - root: root of the original tree
    - data_train
    - data_test
    
    output:
    - internal_nodes: list of the number of internal nodes at each stage of the split
    - accuracy_train: list of the accuracy on the train data at each stage of the split
    - accuracy_test: list of the accuracy on the test data at each stage of the split
    """
    current_list = list_of_parents(root, [], root)
    
    internal_nodes = [number_of_internal_nodes(root)]
    accuracy_train = [calc_accuracy(root, data_train)]
    accuracy_test = [calc_accuracy(root, data_test)]
    node_to_remove = root
    

    while len(root.children) > 0:
 
        max_accuracy = -1
    
        for node in current_list:
            
            temp_children = node.children.copy()
            node.children = []
            #current_accuracy = calc_accuracy(root, data_train)
            current_accuracy = calc_accuracy(root, data_test)
        
            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                node_to_remove = node
        
            node.children = temp_children.copy()
       
        node_to_remove.children = []
        
        internal_nodes.append(number_of_internal_nodes(root))
        # last iteration
        if internal_nodes[-1] == 0:
            # single node
            internal_nodes[-1] = 1
            
        accuracy_train.append(calc_accuracy(root, data_train))
        accuracy_test.append(max_accuracy)
        current_list = list_of_parents(root, [], root)
        
    
    return internal_nodes, accuracy_train, accuracy_test



def print_tree(node):
    """
    prints the tree according to the example in the notebook
    
    Input:
    - node: a node in the decision tree
    
    This function has no return value
    """
    
    print_recursion_tree(node, 0)
    

def print_recursion_tree(node, number_of_tabs):
    
    if node.is_leaf():
        
        num_instances = node.instances
        final_num = {}
        if num_instances[0] == 0:
            final_num[1] = num_instances[1]
        elif num_instances[1] == 0:
            final_num[0] = num_instances[0]
        else:
            final_num = num_instances
        
        print("  " * number_of_tabs + "leaf: [" + str(final_num) + "]")
        return
    
    else:
        print("  " * number_of_tabs + "[X" + str(node.feature) + " <= " + str(node.value) + "],")

    print_recursion_tree(node.children[0], number_of_tabs + 1)
    print_recursion_tree(node.children[1], number_of_tabs + 1)
    
    