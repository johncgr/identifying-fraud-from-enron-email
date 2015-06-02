# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:51:51 2015

@author: john
"""

#!/usr/bin/python

import time
t_beg = time.time()
import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'bonus', 'salary',
                 'exercised_stock_options' ] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
     
for k in data_dict:
    dk = data_dict[k]
    if (dk['from_this_person_to_poi'] == 'NaN' 
      or dk['from_messages'] == 'NaN'
      or dk['from_messages'] == 0):
        dk['prop_to_poi'] = 'NaN'
        
        if (dk['from_poi_to_this_person'] == 'NaN'
          or dk['to_messages'] == 'NaN'
          or dk['to_messages'] == 0 ):
            dk['prop_from_poi'] = 'NaN'               
            continue

    if (dk['from_poi_to_this_person'] == 'NaN'
      or dk['to_messages'] == 'NaN'
      or dk['to_messages'] == 0 ):
        dk['prop_from_poi'] = 'NaN'               
        continue     
                        
    dk['prop_to_poi'] =  ( np.float(dk['from_this_person_to_poi'])
                             / np.float(dk['from_messages']) )
                             
    dk['prop_from_poi'] = ( np.float(dk['from_poi_to_this_person'])
                             / np.float(dk['to_messages']) ) 
                             
for k in data_dict:
    dk = data_dict[k]
    if ( dk['from_this_person_to_poi'] == 'NaN' 
      or dk['from_poi_to_this_person'] == 'NaN'
      or dk['shared_receipt_with_poi'] == 'NaN' ):
        dk['poi_exposure'] = 'NaN'
        continue                          

    dk['poi_exposure'] = ( 1 * dk['prop_from_poi'] 
    + 1 * dk['prop_to_poi']
    + 1 * ( dk['shared_receipt_with_poi']/dk['to_messages'] ) ) 
           
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

### Scale Features in Preperation for k-NN
from sklearn import preprocessing
data = preprocessing.MinMaxScaler().fit_transform(data)

labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB() 

#from sklearn import svm
#clf = svm.LinearSVC(C=100)

#from sklearn import neighbors
#from sklearn.grid_search import GridSearchCV
#parameters = {'n_neighbors':[5,6,7,8], 'weights' : ('distance', 'uniform') }
#svr = neighbors.KNeighborsClassifier(algorithm = 'brute')
#clf = GridSearchCV(svr, parameters)

#from sklearn import tree
#clf = tree.DecisionTreeClassifier(criterion = "entropy", 
#                                  min_samples_split = 2, max_depth = 8,
#                                  splitter = "best")  


from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(n_neighbors = 6, weights = 'distance', 
                                     algorithm = 'brute' )

print(time.time() - t_beg)
 
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)

