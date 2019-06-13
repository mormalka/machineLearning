from numpy import count_nonzero, logical_and, logical_or, concatenate, mean, array_split, poly1d, polyfit, array
from numpy.random import permutation
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt


SVM_DEFAULT_DEGREE = 3
SVM_DEFAULT_GAMMA = 'auto'
SVM_DEFAULT_C = 1.0
ALPHA = 1.5


def prepare_data(data, labels, max_count=None, train_ratio=0.8):
    """
    :param data: a numpy array with the features dataset
    :param labels:  a numpy array with the labels
    :param max_count: max amout of samples to work on. can be used for testing
    :param train_ratio: ratio of samples used for train
    :return: train_data: a numpy array with the features dataset - for train
             train_labels: a numpy array with the labels - for train
             test_data: a numpy array with the features dataset - for test
             test_labels: a numpy array with the features dataset - for test
    """
    
    if max_count:
        data = data[:max_count]
        labels = labels[:max_count]
 
    array_concat = concatenate((data,array([labels]).T),axis=1)
    array_to_shuffle = permutation(array_concat)
    
    split_rows = (int) (train_ratio *array_to_shuffle.shape[0])
    train, test = array_to_shuffle[:split_rows,:],array_to_shuffle[split_rows:,:]
   
    train_data = array (train[:, :-1]) #all but the last
    train_labels = array (train[:, -1]) # for last column
    test_data = array (test[:, :-1]) #all but the last
    test_labels = array (test[:, -1]) # for last column


    return train_data, train_labels, test_data, test_labels


def get_stats(prediction, labels):
    """
    :param prediction: a numpy array with the prediction of the model
    :param labels: a numpy array with the target values (labels)
    :return: tpr: true positive rate
             fpr: false positive rate
             accuracy: accuracy of the model given the predictions
    """
    
    
    positive = count_nonzero(labels)
    negative = labels.shape[0] - positive
    
    tp = sum(logical_and(prediction == 1, labels == 1))
    tn = sum(logical_and(prediction == 0, labels == 0))
    fp = sum(logical_and(prediction == 1, labels == 0))
    fn = sum(logical_and(prediction == 0, labels == 1))
    
          
    tpr = tp / positive
    fpr = fp / negative
    accuracy = (tp + tn)/(tp + fp + tn + fn)


    return tpr, fpr, accuracy


def get_k_fold_stats(folds_array, labels_array, clf):
    """
    :param folds_array: a k-folds arrays based on a dataset with M features and N samples
    :param labels_array: a k-folds labels array based on the same dataset
    :param clf: the configured SVC learner
    :return: mean(tpr), mean(fpr), mean(accuracy) - means across all folds
    """
    #print(folds_array)
    
    tpr = []
    fpr = []
    accuracy = []
    
    for i in range(len(folds_array)):
        
        valid_data = folds_array.pop(0)
        valid_labels = labels_array.pop(0)
        train_data = folds_array
        train_labels =labels_array
        #Fit the SVM model according to the given training data
        clf.fit(concatenate(array(train_data)),concatenate(array(train_labels)))
        #Perform classification on samples in valid_data
        prediction = clf.predict(valid_data)
        tpr_current, fpr_current, accuracy_current = get_stats(prediction,valid_labels)
        
        tpr.append(tpr_current)
        fpr.append(fpr_current)
        accuracy.append(accuracy_current)
        
        folds_array.append(valid_data)        
        labels_array.append(valid_labels)

    return mean(tpr), mean(fpr), mean(accuracy)


def compare_svms(data_array,
                 labels_array,
                 folds_count,
                 kernels_list=('poly', 'poly', 'poly', 'rbf', 'rbf', 'rbf',),
                 kernel_params=({'degree': 2}, {'degree': 3}, {'degree': 4}, {'gamma': 0.005}, {'gamma': 0.05}, {'gamma': 0.5},)):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :param kernels_list: a list of strings defining the SVM kernels
    :param kernel_params: a dictionary with kernel parameters - degree, gamma, c
    :return: svm_df: a dataframe containing the results as described below
    """
    svm_df = pd.DataFrame()
    
    data_folds_array= array_split(data_array,folds_count)
    labels_folds_array = array_split(labels_array,folds_count)
        
    tpr = []
    fpr = []
    accuracy =[]
    
    clf = SVC(gamma = SVM_DEFAULT_GAMMA, C = SVM_DEFAULT_C, degree = SVM_DEFAULT_DEGREE)

    for i in range(len(kernel_params)):
        kernel = kernels_list[i]
        param = kernel_params[i]
       
        clf.set_params(**{'kernel' : kernel, 'gamma' : SVM_DEFAULT_GAMMA, 'C' : SVM_DEFAULT_C, 'degree' : SVM_DEFAULT_DEGREE})
 
        clf.set_params(**param)
           
        tpr_i, fpr_i, accuracy_i = get_k_fold_stats(data_folds_array, labels_folds_array, clf)
        tpr.append(tpr_i)
        fpr.append(fpr_i)
        accuracy.append(accuracy_i)
        
        
    svm_df['kernel'] = kernels_list
    svm_df['kernel_params'] = kernel_params
    svm_df['tpr'] = tpr
    svm_df['fpr'] = fpr
    svm_df['accuracy'] = accuracy


    return svm_df


def get_most_accurate_kernel(accuracy):
    """
    :return: integer representing the row number of the most accurate kernel
    """
    accuracy_list = accuracy.tolist()
    
    best_kernel = accuracy_list.index(max(accuracy))
    return best_kernel


def get_kernel_with_highest_score(scores):
    """
    :return: integer representing the row number of the kernel with the highest score
    """
    scores_list = scores.tolist()
    
    best_kernel = scores_list.index(max(scores))
    return best_kernel
   


def plot_roc_curve_with_score(df, alpha_slope=1.5):
    """
    :param df: a dataframe containing the results of compare_svms
    :param alpha_slope: alpha parameter for plotting the linear score line
    :return:
    """
    x = df.fpr.tolist()
    y = df.tpr.tolist()
    
    #Calculating the index of kernel with the highest score 
    best_kernel_idx = get_kernel_with_highest_score(df['score'])
    
    #Finds b value by exist point  values x,y
    b = y[best_kernel_idx] - (alpha_slope*x[best_kernel_idx])
    #Create a polynomial with the given coefficients
    line = poly1d([alpha_slope,b])
    line_x = [0,5]
    line_y = line([0,5])

    plt.plot(line_x, line_y, color = 'b')
    plt.plot(x, y, 'ro', ms=5, color = 'r')
    plt.title('ROC curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.show()

def duplicate (shape, value):
    
    res = []
    
    for i in range (shape):
        res.append(value)
    return res 
    

def evaluate_c_param(data_array, labels_array, folds_count, params, kernel):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :param kernel_params: a dictionary with best kerenel params
    :return: res: a dataframe containing the results for the different c values. columns similar to `compare_svms`
    """

    res = pd.DataFrame()
    kernel_params = []
    
    for i in [1,0,-1,-2,-3,-4]:
        
         for j in [1,2,3]:
                current_params = params.copy()
                current_params['C'] =((10**i) * ((j)/3))
                kernel_params.append(current_params)
    ###########################################################################fix
   
    kernel_list =  duplicate(len(kernel_params),kernel)
    
    res = compare_svms(data_array, labels_array, folds_count, kernel_list,kernel_params)
    
    return res


def get_test_set_performance(train_data, train_labels, test_data, test_labels,params, kernel):
    """
    :param train_data: a numpy array with the features dataset - train
    :param train_labels: a numpy array with the labels - train

    :param test_data: a numpy array with the features dataset - test
    :param test_labels: a numpy array with the labels - test
    :return: kernel_type: the chosen kernel type (either 'poly' or 'rbf')
             kernel_params: a dictionary with the chosen kernel's parameters - c value, gamma or degree
             clf: the SVM leaner that was built from the parameters
             tpr: tpr on the test dataset
             fpr: fpr on the test dataset
             accuracy: accuracy of the model on the test dataset
    """

    kernel_type = kernel
    kernel_params = params
   
    clf = SVC(class_weight = 'balanced')  
    # TODO: set the right kernel
    
    clf.set_params(**{'kernel' : kernel, 'gamma' : SVM_DEFAULT_GAMMA, 'C' : SVM_DEFAULT_C, 'degree' : SVM_DEFAULT_DEGREE})
    clf.set_params(**params)
    
    clf.fit(train_data,train_labels)
    
    prediction = clf.predict(test_data)
    tpr, fpr, accuracy = get_stats(prediction,test_labels)

    return kernel_type, kernel_params, clf, tpr, fpr, accuracy
