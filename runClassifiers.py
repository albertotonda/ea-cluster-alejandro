# Script that makes use of more advanced feature selection techniques
# by Alberto Tonda, 2017

import copy
import datetime
import graphviz
import logging
import numpy as np
import os
import sys
import pandas as pd 

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier

from sklearn.multiclass import OneVsOneClassifier 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

# used for normalization
from sklearn.preprocessing import  Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# used for cross-validation
from sklearn.model_selection import StratifiedKFold

# this is an incredibly useful function
from pandas import read_csv

def loadDataset() :
	
	# data used for the predictions
	dfData = read_csv("./data/data_0.csv", header=None, sep=',')
	dfLabels = read_csv("./data/labels.csv", header=None)
	biomarkers = read_csv("./data/features_0.csv", header=None)

	return dfData.values, dfLabels.values.ravel(), biomarkers.values.ravel() # to have it in the format that the classifiers like


def runFeatureReduce(X,y) :
	
	# a few hard-coded values
	numberOfFolds = 10
	
	# list of classifiers, selected on the basis of our previous paper "
	classifierList = [
		
			[GradientBoostingClassifier(n_estimators=300), "GradientBoostingClassifier(n_estimators=300)"],
			[RandomForestClassifier(n_estimators=300), "RandomForestClassifier(n_estimators=300)"],
			[LogisticRegression(solver='lbfgs'), "LogisticRegression"],
			[PassiveAggressiveClassifier(),"PassiveAggressiveClassifier"],
			[SGDClassifier(), "SGDClassifier"],
			[SVC(kernel='linear'), "SVC(linear)"],
			[RidgeClassifier(), "RidgeClassifier"],
			[BaggingClassifier(n_estimators=300, n_jobs=1), "BaggingClassifier(n_estimators=300)"],
			# ensemble
			

			]
	
	# this is just a hack to check a few things
	#classifierList = [
	#		[RandomForestClassifier(), "RandomForestClassifier"]
	#		]


	# prepare folds
	skf = StratifiedKFold(n_splits=numberOfFolds, shuffle=True)
	indexes = [ (training, test) for training, test in skf.split(X, y) ]
	globalPerformance=[]
	for originalClassifier, classifierName in classifierList :

		classifierPerformance = []

		
		for train_index, test_index in indexes :
			
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			
			# let's normalize, anyway
			# MinMaxScaler StandardScaler Normalizer
			scaler = StandardScaler()
			X_train = scaler.fit_transform(X_train)
			X_test = scaler.transform(X_test)

			
			classifier = copy.deepcopy(originalClassifier)
			classifier.fit(X_train, y_train)
			scoreTraining = classifier.score(X_train, y_train)
			scoreTest = classifier.score(X_test, y_test)
					
			
			#print("\ttraining: %.4f, test: %.4f" % (scoreTraining, scoreTest))
			classifierPerformance.append( scoreTest )

		line ="%s \t %.4f \t %.4f " % (classifierName, np.mean(classifierPerformance), np.std(classifierPerformance))
		globalPerformance.append(np.mean(classifierPerformance)-np.std(classifierPerformance))
		#print(line)
	
	#print(np.mean(globalPerformance))
	return np.mean(globalPerformance)

