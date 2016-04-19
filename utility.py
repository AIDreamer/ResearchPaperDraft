## Name: Son Pham
## Research for professor King
## Preprocess the data

import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, metrics, linear_model
from random import randrange, choice
import pandas as pd
from ggplot import *

import matplotlib.colors as mcolors

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def preprocess_breast_cancer_data(features, info):

    # Load the data
    train_data = np.loadtxt(features)
    train_info = np.loadtxt(info)
    train_labels = [train_info[x][0] for x in range(len(train_info))]
    N = len(train_labels)

    # Reprocess the data into 1 (Malignant) and 0 (Non-malignant) format
    train_labels = [[1] if x == 1.0 else [0] for x in train_labels]
    train_labels = np.array(train_labels)

    return train_data, train_labels

def sort_out_data(train_data, train_labels):

    N = len(train_labels)
    
    # Take out all positive instances
    pos_data = [train_data[i] for i in range(N) if train_labels[i] == 1]
    neg_data = [train_data[i] for i in range(N) if train_labels[i] == 0]
    return pos_data, neg_data


def find_k_nearest_neighbors(k, pos_data):

    # Number of instances
    num_pos = len(pos_data)
    num_attributes = len(pos_data[0])

    # Figure out all the k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(pos_data)
    distances, indices = nbrs.kneighbors(pos_data)

    # Create a table of all the nearest pieces of data
    near_data = np.zeros((num_pos, k, num_attributes))
    for i in range(num_pos):
        for j in range(k):
            near_data[i][j] = pos_data[indices[i][j]]

    return near_data

def calculate_sigma_table(near_data):

    #number of instances
    num_pos = len(near_data)
    num_attributes = len(near_data[0][0])

    # Create a table of standard deviation for the nearest neighbours.
    sigma_table = np.zeros((num_pos, num_attributes))
                           
    # Calculate all those standard deviation
    for i in range(num_pos):
                           
        # Grab the data
        data = np.array(near_data[i])
        data = np.rot90(data)
        data = np.flipud(data)
                           
        # Calculate the standard deviation
        sigma_table[i] = np.std(data, axis = 1)

    return sigma_table
    

'''
    Using Gaussian distribution to augment a bunch of data
'''
def augment_data(data, sigma):

    new_data = np.copy(data)
    l = len(new_data)
    for i in range(l):
        if sigma[i] > 0:
            new_data[i] = np.random.normal(new_data[i],sigma[i])
    return new_data

def generate_new_data_points(num_new, pos_data, sigma_table):

    #number of instances
    num_pos = len(pos_data)
    num_attributes = len(pos_data[0])
    new_data = np.zeros((num_new, num_attributes))

    # Generate num_new random points
    random_point = np.random.randint(num_pos, size = num_new)

    for i in range(num_new):
        
        # Pick the random piece of data
        temp_point = random_point[i]
        
        # Generate new data using the method
        new_data[i] = augment_data(pos_data[temp_point], sigma_table[temp_point])

    return new_data

'''
    Calculate vector-pushed new average based on new data
'''
def generate_vector_push_points(near_data, pos_data):
    num_pos = len(pos_data)
    num_attr = len(pos_data[0])
    num_near = len(near_data[0])
    new_points = [0] * num_pos

    for i in range(num_pos):
        pos_point = pos_data[i]
        all_points = np.array(near_data[i])
        avg_point = np.mean(all_points, axis = 0) # Center of all points
        # Calculate distance of all_points to center point
        all_dists = [0] * num_near
        for j in range(num_near):
            all_dists[j] = np.array(np.linalg.norm(all_points[j] - avg_point))
        avg_dist = sum(all_dists) / len(all_dists)
        
        # Caluclate partial distance with sigmoid function
        fraction = avg_dist / (avg_dist + np.exp(avg_dist - pos_point))

        # Calculate the new point based on fractional distance
        new_points[i] = avg_point + fraction * (pos_data[i] - avg_point)

    new_points = np.array(new_points)

    return new_points

def vector_pushed_gaussian_augment(k, pos_data, num_new, c):
    near_data = find_k_nearest_neighbors(k, pos_data)
    sigma_table = calculate_sigma_table(near_data)
    push_data = genetate_vector_push_points(near_data, pos_data)
    new_data = generate_new_data_points(num_new, push_data, c * sigma_table)
    return new_data

def knn_gaussian_augment(k, pos_data, num_new, c):
    near_data = find_k_nearest_neighbors(k, pos_data)
    sigma_table = calculate_sigma_table(near_data)
    new_data = generate_new_data_points(num_new, pos_data, c * sigma_table)
    return new_data

def gaussian_oversample(train_data, train_labels, num_neighbors, percent_new, c):
    
    # Oversampling the positive data
    pos_data, neg_data = sort_out_data(train_data, train_labels)
    num_new = percent_new * len(pos_data) // 100
    if (num_neighbors == 0): num_neibhbors = len(pos_data)
    new_data = knn_gaussian_augment(num_neighbors, pos_data, num_new, c)

    # Append the newly created positive data to the set.
    new_train_data = np.concatenate((train_data, new_data), axis=0)
    new_train_labels = np.concatenate((train_labels, [1] * num_new))
    
    return new_train_data, new_train_labels

def SMOTE_oversample(train_data, train_labels, num_neighbors, percent_new):
    
    # Oversampling the positive data
    pos_data, neg_data = sort_out_data(train_data, train_labels)
    if (num_neighbors == 0): num_neighbors = len(pos_data)
    new_data = SMOTE(pos_data, percent_new, num_neighbors)
    num_new = len(new_data)
    # Append the newly created positive data to the set.
    new_train_data = np.concatenate((train_data, new_data), axis=0)
    new_train_labels = np.concatenate((train_labels, [1] * num_new))
    
    return new_train_data, new_train_labels

def normalize_AGO_oversample(train_data, train_labels, sigma_array, percent_new, c):

    # Identify pos_data
    pos_data, neg_data = sort_out_data(train_data, train_labels)

    # Oversample the positive data
    num_pos = len(pos_data)
    num_new = percent_new * len(pos_data) // 100
    sigma_table = np.tile([sigma_array], (num_pos, 1))
    new_data = generate_new_data_points(num_new, pos_data, c * sigma_table)

    # Append the newly created positive data to the set
    new_train_data = np.concatenate((train_data, new_data), axis=0)
    new_train_labels = np.concatenate((train_labels, [1] * num_new))

    return new_train_data, new_train_labels

def divide_data_train_test_ratio(train_data, train_labels, den, num):

    # Number of attributes
    num_attr = len(train_data[0])

    # Attach the labels to the data (for the shake of convenient shuffle)
    train_data = np.append(train_data, train_labels, axis=1)

    # Partition the training data into specified ratio
    trainingN = len(train_data) * den // num

    # Shuffle the train_data
    np.random.shuffle(train_data)
    new_test_data = train_data[trainingN:]
    new_train_data = train_data[:trainingN]
    new_test_labels = np.array([new_test_data[x][num_attr] for x in range(len(new_test_data))])
    new_train_labels = np.array([new_train_data[x][num_attr] for x in range(len(new_train_data))])
    new_train_data = np.delete(new_train_data, num_attr, 1)
    new_test_data = np.delete(new_test_data, num_attr, 1)

    return new_train_data, new_train_labels, new_test_data, new_test_labels


def divide_data_k_folds(train_data, train_labels, K):

    # Number of attributes
    num_attr = len(train_data[0])
    
    # Attach the labels to the data (for the shake of convenient shuffle)
    train_data = np.append(train_data, train_labels, axis=1)

    # Partition the training data into K parts
    N = len(train_labels);
    smallN = N // K

    # Shuffle the train_data and create K parts
    np.random.shuffle(train_data)
    data_parts = [0] * K;

    for i in range(K):
        begin = smallN * i
        end = smallN * (i+1)
        if (i == K-1):
            data_parts[i] = train_data[begin : ]
        else:
            data_parts[i] = train_data[begin : end]

    # Create K set of training and testing
    all_test_datasets = [0] * K
    all_train_datasets = [0] * K

    for i in range(K):
        all_test_datasets[i] = data_parts[i]
        all_train_datasets[i] = np.empty( shape = (0, num_attr+1))
        for j in range(K):
            if (j != i):
                all_train_datasets[i] = np.concatenate((all_train_datasets[i], data_parts[j]), axis = 0)
                
    # Clean up data and create all the labels
    all_test_labels = [0] * K
    all_train_labels = [0] * K

    for i in range(K):
        test_set = all_test_datasets[i]
        train_set = all_train_datasets[i]
        
        # Grab labels
        all_test_labels[i] = np.array([test_set[x][num_attr] for x in range(len(test_set))])
        all_train_labels[i] = np.array([train_set[x][num_attr] for x in range(len(train_set))])

        # Clean up
        all_test_datasets[i] = np.delete(all_test_datasets[i], num_attr, 1)
        all_train_datasets[i] = np.delete(all_train_datasets[i], num_attr, 1)

    return all_train_datasets, all_test_datasets, all_train_labels, all_test_labels

def logit_predict(new_train_data, new_train_labels, test_data, test_labels):
    
    # Create a classifier: a logistic classifier
    classifier = linear_model.LogisticRegression()

    # We learn the digits on the first half of the digits
    classifier.fit(new_train_data, new_train_labels)
    
    # Now predict the value of the digit on the second half:
    expected = test_labels
    predicted = classifier.predict_proba(test_data)
    
    return predicted

def SVM_predict(new_train_data, new_train_labels, test_data, test_labels):
    
    # Create a classifier: a logistic classifier
    classifier = linear_model.LogisticRegression()

    # We learn the digits on the first half of the digits
    classifier.fit(new_train_data, new_train_labels)
    
    # Now predict the value of the digit on the second half:
    expected = test_labels
    predicted = classifier.predict_proba(test_data)
    
    return predicted

def random_forest_predict(new_train_data, new_train_labels, test_data, test_labels, n = 500):

    # Create a classifier: a random forest
    classifier = RandomForestClassifier(n_estimators = n)

    # We learn the digits on the first half of the digits
    classifier.fit(new_train_data, new_train_labels)
    
    # Now predict the value of the digit on the second half:
    expected = test_labels
    predicted = classifier.predict_proba(test_data)

    return predicted

def Gaussian_NB_predict(new_train_data, new_train_labels, test_data, test_labels):

    # Create a classifier: a Gaussian Naive Bayesian
    classifier = GaussianNB()

    # We learn the digits on the first half of the digits
    classifier.fit(new_train_data, new_train_labels)
    
    # Now predict the value of the digit on the second half:
    expected = test_labels
    predicted = classifier.predict_proba(test_data)

    return predicted

def kNN_predict(new_train_data, new_train_labels, test_data, test_labels, n_neighbors = 10):
    
    # Create a classifier: k-nearest neighbours
    classifier = KNeighborsClassifier(n_neighbors)

    # We learn the digits on the first half of the digits
    classifier.fit(new_train_data, new_train_labels)

    # Now predict the value of the digit on the second half:
    expected = test_labels
    predicted = classifier.predict_proba(test_data)

    return predicted

def ada_boost_predict(new_train_data, new_train_labels, test_data, test_labels, base_est = "tree", n = 50):

    # Create a classifier: AdaBoost classifier
    if base_est == "tree":
        base = DecisionTreeClassifier(max_depth=5)
    classifier = AdaBoostClassifier(base_estimator = base, n_estimators = n)

    # We learn the digits on the first half of the digits
    classifier.fit(new_train_data, new_train_labels)

    # Now predict the value of the digit on the second half:
    expected = test_labels
    predicted = classifier.predict_proba(test_data)

    return predicted

def tree_predict(new_train_data, new_train_labels, test_data, test_labels, depth = 10):

    # Create a classifier: Decision tree
    classifier = DecisionTreeClassifier(max_depth=depth)
    
    # We learn the digits on the first half of the digits
    classifier.fit(new_train_data, new_train_labels)

    # Now predict the value of the digit on the second half:
    expected = test_labels
    predicted = classifier.predict_proba(test_data)

    return predicted

def SVMRBF_predict(new_train_data, new_train_labels, test_data, test_labels, depth = 10):

    # Create a classifier: Decision tree
    classifier = svm.SVC(probability=True)
    
    # We learn the digits on the first half of the digits
    classifier.fit(new_train_data, new_train_labels)

    # Now predict the value of the digit on the second half:
    expected = test_labels
    predicted = classifier.predict_proba(test_data)

    return predicted

def predict(train_data, train_labels, test_data, test_labels, method):
    if method == "Logistic Regression":
        return logit_predict(train_data, train_labels, test_data, test_labels)
    elif method == "kNN":
        return kNN_predict(train_data, train_labels, test_data, test_labels)
    elif method == "Decision Tree":
        return tree_predict(train_data, train_labels, test_data, test_labels)
    elif method == "Random Forest":
        return random_forest_predict(train_data, train_labels, test_data, test_labels)
    elif method == "Naive Bayes":
        return Gaussian_NB_predict(train_data, train_labels, test_data, test_labels)
    elif method == "AdaBoost - Tree":
        return ada_boost_predict(train_data, train_labels, test_data, test_labels)
    elif method == "SVM":
        return SVMRBF_predict(train_data, train_labels, test_data, test_labels)

def calculateAUC(test_labels, prediction):
    fpr, tpr, _ = metrics.roc_curve(test_labels, prediction)
    return metrics.auc(fpr, tpr)

def drawAUC(test_labels, prediction):
    fpr, tpr, _ = metrics.roc_curve(test_labels, prediction)

    df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
        
    auc = metrics.auc(fpr,tpr)
    
    my_plot = ggplot(df, aes(x='fpr', ymin=0, ymax='tpr')) +\
        geom_area(alpha=0.0) +\
        geom_line(aes(y='tpr')) +\
        ggtitle("ROC Curve w/ AUC=%s" % str(auc))
    print my_plot
    return my_plot

def SMOTE(T, N, k):
    
    """
    Returns (N/100) * n_minority_samples synthetic minority samples.

    Parameters
    ----------
    T : array-like, shape = [n_minority_samples, n_features]
        Holds the minority samples
    N : percetange of new synthetic samples: 
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours. 

    Returns
    -------
    S : array, shape = [(N/100) * n_minority_samples, n_features]
    """

    T = np.array(T)
    
    n_minority_samples, n_features = T.shape
    
    if N < 100:
        #create synthetic samples only for a subset of T.
        #TODO: select random minortiy samples
        N = 100
        pass

    if (N % 100) != 0:
        raise ValueError("N must be < 100 or multiple of 100")
    
    N = N/100
    n_synthetic_samples = N * n_minority_samples
    S = np.zeros(shape=(n_synthetic_samples, n_features))
    
    #Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(T)
    
    #Calculate synthetic samples
    for i in xrange(n_minority_samples):
        nn = neigh.kneighbors(T[i], return_distance=False)
        for n in xrange(N):
            nn_index = choice(nn[0])
            #NOTE: nn includes T[i], we don't want to select it 
            while nn_index == i:
                nn_index = choice(nn[0])
                
            dif = T[nn_index] - T[i]
            gap = np.random.random()
            S[n + i * N, :] = T[i,:] + gap * dif[:]
    
    return S
