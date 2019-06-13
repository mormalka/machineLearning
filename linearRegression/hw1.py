import numpy as np
import itertools
np.random.seed(42)

def preprocess(X, y):
    
    X_mean = X.mean(axis = 0)
    X_min = X.min(axis = 0)
    X_max = X.max(axis = 0)
    
    X = (X - X_mean) / (X_max - X_min) # perform mean normalization on the features
    y = (y - y.mean()) / (y.max() - y.min()) # perform mean normalization on the target values
    
    return X, y

def compute_cost(X, y, theta):
    
    hFunction = X.dot(theta) 
    sq_error= np.square(np.subtract(hFunction, y)) # computes the squared difference
    J = np.sum(sq_error) / (2 * len(y))
    
    # The cost associated with the current set of parameters
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
  
    J_history = [] 
    scalar = alpha / len(y) 
    
    for i in range (num_iters):
    
        hFunction = X.dot(theta) 
        sigma = ((np.subtract(hFunction, y)).dot(X)) * scalar #calculating sigma multiple by scalar
        theta = theta - sigma # calculating new theta
        J_history.append(compute_cost(X, y, theta))
       
    return theta, J_history

def pinv(X, y):

    pinv_theta = (np.linalg.inv(X.T.dot(X))).dot(X.T)
    pinv_theta = pinv_theta.dot(y)
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    
    J_history = [] 
    scalar = alpha / len(y)
    
    for i in range (num_iters):
        
        hFunction = X.dot(theta)
        sigma = ((np.subtract(hFunction, y)).dot(X)) * scalar # calculating sigma multiple by scalar
        theta = theta - sigma # calculating new theta
        J_history.append(compute_cost(X, y, theta))
        
        # Stop if improvement of the loss value is smaller than 1e-8
        if (i > 0) and (J_history[i-1]-J_history[i] < 1e-8):
            return theta, J_history
        
    return theta, J_history

def find_best_alpha(X, y, iterations):
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}
    theta = np.random.random(size=2)
    
    # Creating dictionary with alpha as the key and the final loss as the value.
    for i in range(len(alphas)):
        temp, loss = efficient_gradient_descent(X, y, theta, alphas[i],iterations)
        alpha_dict.update( { alphas[i] : loss[-1] } )
    
    return alpha_dict

def generate_triplets(X):
    
    triplets = list(itertools.combinations(X, 3))
   
    return triplets

def find_best_triplet(df, triplets, alpha, num_iter):
    
    min_cost_triplet = []
    theta = np.random.random(size = 4)
    
    for a in triplets:
        y = np.array(df['price'])
        array = [a[0],a[1],a[2]]
        X = np.array( df[array] )
        # preprocess the data and obtain a array containing the columns corresponding to the triplet
        X,y =preprocess(X,y) #bias trick
        X = np.insert(X, 0, 1, axis=1)
        current_theta, loss = efficient_gradient_descent(X, y, theta, alpha, num_iter)
        min_cost_triplet.append(loss[-1])
        
                                    
    pos = min_cost_triplet.index(min(min_cost_triplet))
    best_triplet = list(triplets[pos])   
    
    return best_triplet
                         