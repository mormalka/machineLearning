import numpy as np
np.random.seed(42)

####################################################################################################
#                                            Part A
####################################################################################################

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.same_class = self.dataset[self.dataset[:,-1] == class_value] # counts how many instances have the class value as the current object
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.same_class.shape[0] / self.dataset.shape[0]
    
    def get_mean_std(self):
        """
        Returns the mean and std of each feature according to the dataset distribution and the formula we learned in class.
        """
        mean_temp = (1 / self.same_class.shape[0]) * (np.sum(self.same_class[:,0]))
        mean_humid = (1 / self.same_class.shape[0]) * (np.sum(self.same_class[:,1]))
        
        std_temp = np.sqrt((1 / self.same_class.shape[0]) * np.sum(np.square(self.same_class[:,0] - mean_temp)))
        std_humid = np.sqrt((1 / self.same_class.shape[0]) * np.sum(np.square(self.same_class[:,1] - mean_humid)))
        
        return mean_temp, mean_humid, std_temp, std_humid
    
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        mean_temp, mean_humid, std_temp, std_humid = self.get_mean_std()
        
        return normal_pdf(x[0], mean_temp, std_temp) * normal_pdf(x[1], mean_humid, std_humid)
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()
    
class MultiNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.same_class = self.dataset[self.dataset[:,-1] == class_value] # counts how many instances have the class value as the current object
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.same_class.shape[0] / self.dataset.shape[0]
    
    def get_mean_cov(self):
        """
        Returns the mean and covariance of each feature according to the dataset distribution and the formula we learned in class.
        """
        mean_temp = (1 / self.same_class.shape[0]) * (np.sum(self.same_class[:,0]))
        mean_humid = (1 / self.same_class.shape[0]) * (np.sum(self.same_class[:,1]))
        
        mean = np.array([mean_temp, mean_humid])
        
        cov = np.cov(self.same_class[:, :-1].T)
        
        return mean, cov
        
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        x = x[:-1]
        mean, cov = self.get_mean_cov()
        
        return multi_normal_pdf(x, mean, cov)
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()
    
    

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    std_square = np.square(std)
    power_of_e = - ((np.square(x - mean)) / (2 * std_square))
    fraction = 1 / np.sqrt (2 * np.pi * std_square)
    
    return fraction * (np.power(np.e, power_of_e))
    
def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variante normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    fraction = ((2 * np.pi) ** -(len(mean) / 2)) * (np.linalg.det(cov) ** -0.5)
    power_of_e = -0.5 * np.matmul(np.matmul((x - mean).T, np.linalg.inv(cov)), (x - mean))
    
    return fraction * (np.power(np.e, power_of_e))
    
    
####################################################################################################
#                                            Part B
####################################################################################################
EPSILLON = 1e-6 # == 0.000001 It could happen that a certain value will only occur in the test set.
                # In case such a thing occur the probability for that value will EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with la place smoothing.
        
        Input
        - dataset: The dataset from which to compute the probabilites (Numpy Array).
        - class_value : Compute the relevant parameters only for instances from the given class.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.same_class = self.dataset[self.dataset[:,-1] == class_value] # counts how many instances have the class value as the current object
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.same_class.shape[0] / self.dataset.shape[0]
    
    def get_features(self, feature_class, feature_data):
        """
        Returns a dictionary of the every feature value as a key and how many instances of the same value are there as a value.
        """
        key, value = np.unique(feature_class, return_counts=True)
        feature_dict_class = dict(zip(key, value))
        
        key, value = np.unique(feature_data, return_counts=True)
        num_attributes = len(dict(zip(key, value)))
        
        return feature_dict_class, num_attributes
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = 1
        x = x[:-1]
        
        Ni = self.same_class.shape[0] # number of training instances with current class
        
        for i in range(len(x)):
            feature_class, num_attributes = self.get_features(self.same_class[:,i], self.dataset[:,i])
            if x[i] in feature_class:
                Nij = feature_class[x[i]]
                temp = (Nij + 1) / ( Ni + num_attributes)
            else:
                temp = EPSILLON
            
            likelihood *= temp
            
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        post_current_class = self.get_instance_likelihood(x) * self.get_prior()
        temp_object = None
        if self.class_value == 0:
            temp_object = DiscreteNBClassDistribution(self.dataset, 1)
        else:
            temp_object = DiscreteNBClassDistribution(self.dataset, 0)
            
        post_other_class = temp_object.get_instance_likelihood(x) * temp_object.get_prior()
        
        return post_current_class / (post_current_class + post_other_class) 

    
####################################################################################################
#                                            General
####################################################################################################            
class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a postreiori classifier. 
        This class will hold 2 class distribution, one for class 0 and one for class 1, and will predicit and instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
        
        Input
            - An instance to predict.
            
        Output
            - 0 if the posterior probability of class 0 is higher 1 otherwise.
        """
        
        post_0 = self.ccd0.get_instance_posterior(x)
        post_1 = self.ccd1.get_instance_posterior(x)
        
        if post_0 > post_1:
            return 0
        else:
            return 1
    
def compute_accuracy(testset, map_classifier):
    """
    Compute the accuracy of a given a testset and using a map classifier object.
    
    Input
        - testset: The test for which to compute the accuracy (Numpy array).
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / #testset size
    """
    
    correct_pred = 0
    for instance in testset :
        pred = map_classifier.predict(instance)
        if pred == instance[-1]:
            correct_pred += 1
    
            
    return (correct_pred / testset.shape[0])







