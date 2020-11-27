#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:25:54 2020

@author: julian-tejada
@author: gabesness
"""

# from os import listdir
# from os.path import isfile, join
import argparse
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import pylab as pl
import matplotlib.pyplot as plt

# datasets tem tamanhos diferentes

"""
attributes (in order of appearance in the csv file):

ATTR                        INTERVAL
gaze:                       gaze_0_x to gaze_angle_y
eye landmarks 2D:           eye_lmk_x_0 to eye_lmk_y_55
eye landmarks 3D:           eye_lmk_X_0 to eye_lmk_Z_55
head pose (no rotation):    pose_Tx to pose_Tz
head pose (rotation):       pose_Rx to pose_Rz
face landmarks 2D:          x_0 to y_67
face landmarks 3D:          X_0 to Z_67
poses 1:                    p_scale to p_ty
poses 2:                    p_0 a p_33
facs:                       AU01_r to AU45_c

"""


# Create list with paths for the csv files

# csv_files = [f for f in listdir('../csv') if isfile(join('../csv', f))]
csv_files = ['../csv/anger.csv', '../csv/deboche.csv', '../csv/disgust.csv', '../csv/fear.csv', '../csv/happiness.csv', '../csv/neutral.csv', '../csv/sadness.csv', '../csv/surprise.csv']



def create_lists(n):

    # Initialize a dictionary with empty lists as placeholders for the DataFrames
    # Each list is associated to one aspect of the csv files and has a unique prime number key
    # With this method, combinations of aspects can be expressed as the product of their prime numbers
    # The dictionary will contain lists of DataFrames, which will be later concatenated

    aspects = {
    2: [],   # gaze
    3: [],   # eye_2D
    5: [],   # eye_3D
    7: [],   # hpose_t
    11: [],  # hpose_r
    13: [],  # face_2D
    17: [],  # face_3D
    19: [],  # poses_1
    23: [],  # poses_2
    29: []   # facs
    }

    # Define a dicionary containing the intervals of the csv files
    # Notice it has matching keys with 'aspects', so that they can be linked together in the loop

    intervals = {
    2: ['gaze_0_x', 'gaze_angle_y'],        # gaze
    3: ['eye_lmk_x_0', 'eye_lmk_y_55'],     # eye_2D
    5: ['eye_lmk_X_0', 'eye_lmk_Z_55'],     # eye_3D
    7: ['pose_Tx', 'pose_Tz'],              # hpose_t
    11: ['pose_Rx', 'pose_Rz'],             # hpose_r
    13: ['x_0', 'y_67'],                    # face_2D
    17: ['X_0', 'Z_67'],                    # face_3D
    19: ['p_scale', 'p_ty'],                # poses_1
    23: ['p_0', 'p_33'],                    # poses_2
    29: ['AU01_r', 'AU45_c']                # facs
    }

    strings = {
    2: "gaze",
    3: "eye_2D",
    5: "eye_3D",
    7: "head pose (no rotation)",
    11: "head pose (w/ rotation)",
    13: "face_2D",
    17: "face_3D",
    19: "poses_1",
    23: "poses_2",
    29: "facs"
    }

    # Initialize empty list of labels, which will later be a DataFrame

    labels = []
    selection = ""

    for i in aspects.keys():
        if n % i == 0:
            selection += strings[i] + "; "
            start = intervals[i][0]
            end = intervals[i][1]
            for j in range(len(csv_files)):
                temp = pd.read_csv(csv_files[j])
                emotion = temp.loc[:, start : end]
                aspects[i].append(emotion)
            aspects[i] = pd.concat(aspects[i])
        else:
            aspects[i] = pd.DataFrame()
    
    for i in range(len(csv_files)):
        temp = pd.read_csv(csv_files[i])
        temp["class"] = i
        tempClass = temp[["class"]]
        labels.append(tempClass)
    labels = pd.concat(labels)

    #print(selection)    
    
    return aspects, labels, selection

# alinhar classes com dataframes em aspects

def training_and_test(n):
    aspects, labels, selection = create_lists(n)
    dfs = [x for x in aspects.values()]
    full_df = pd.concat([labels, pd.concat(dfs, axis=1)], axis=1)

    sampled_set = full_df.sample(len(full_df))
    training_set = sampled_set[0 : int(len(sampled_set)*0.8)]
    test_set = sampled_set[len(training_set)+1 : len(sampled_set)]

    print(selection)

    return training_set, test_set, selection


def classifier(n, k, c):
    # Separate training and test data from the labels
    training_set, test_set, selection = training_and_test(n)
    training = training_set.iloc[:, 1:]
    training_labels = training_set[["class"]]

    test = test_set.iloc[:, 1:]
    test_labels = test_set[["class"]]

    npar_train = training.to_numpy()
    npar_trainlabs = training_labels.to_numpy()

    npar_predict = test.to_numpy()
    npar_predictlabs = test_labels.to_numpy()
    
    clf = SVC(kernel=k, C=c, probability=True)
    classifier = clf.fit(npar_train, npar_trainlabs.flatten())
    
    predictions = clf.predict(npar_predict)
    probabilities = clf.predict_proba(npar_predict)
    
    cm = confusion_matrix(test_labels, predictions, normalize="true")

    pl.matshow(cm)
    pl.title("Heat map of: " + selection)
    #pl.colorbar()
    pl.show()
    
    return cm
    

parser = argparse.ArgumentParser(description="Train a classifier to predict and classify emotions")

# Positional (mandatory arguments)
# parser.add_argument("n", type=int, help="int (or product of ints) of aspect(s) to be trained")
parser.add_argument("-l", "--list", nargs='+', help="list of attributes", required=True)

# Optional arguments
parser.add_argument("-k", "--kernel", default='linear', help="set kernel of the SVC (default is set to linear)")
parser.add_argument("-c", "--C", type=float, default=10, help="set C value of the SVC (default is set to 10)")

args = parser.parse_args()


#print(create_lists(2))
#print(training_and_test(17))

#classifier(args.n, args.kernel, args.C)

for m in args.list:
    m = int(m)
    classifier(m, args.kernel, args.C)
    