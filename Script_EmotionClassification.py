#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script adapted from  http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/
Created on Fri Jul 24 10:56:40 2020

@author: Julian Tejada

"""


import sys
import cv2
import glob
import random
import math
import numpy as np
import pandas as pd
import ast
import csv
import dlib
import itertools
import pylab as pl
from joblib import dump, load
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix




emotions = [ "disgust", "happiness","surprise", "anger", "sadness", "neutral", "fear"] #Emotion list
# emotions = ["neutral", "disgust" ] #Emotion list

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
clf = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel

data = {} #Make dictionary for all values
def get_files(): #Define function to get file list, randomly shuffle it and split 80/20
    prediction_files_names  = glob.glob("%s/*.png" %(sys.argv[1]))
    # training_files_names  = glob.glob("%s/%s/*.png" %(sys.argv[2], emotion))
    # training = training_files_names
    prediction = prediction_files_names
    return prediction
def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        if xlist[26] == xlist[29]: # Si la coordenada x del conjunto son las mismas, el ángulo es 0,  evitamos el error 'divide by 0' en la función
            anglenose = 0
        else:
            anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi)
        if anglenose < 0:
            anglenose += 90
        else:
            anglenose -= 90
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
        
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"
def make_sets():
    # training_data = []
    # training_labels = []
    prediction_data = []
    prediction_labels = []
    prediction_names = []
    names_data = []
    prediction = get_files()
    
    # for emotion in emotions:
    #     print(" working on %s" %emotion)
    #     
    #     names_data.append(names)
        
    #     #Append data to training and prediction list, and generate labels 0-7
    #     for item in training:
    #         image = cv2.imread(item) #open image
    #         print(" working on image %s" %item)
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
    #         clahe_image = clahe.apply(gray)
    #         get_landmarks(clahe_image)
    #         if data['landmarks_vectorised'] == "error":
    #             print("no face detected on this one")
    #         else:
    #             training_data.append(data['landmarks_vectorised']) #append image array to training data list
    #             training_labels.append(emotions.index(emotion))
    prediction_names.append(prediction)           
    for item in prediction:
        image = cv2.imread(item)
        #print("Image %s" %item)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe_image = clahe.apply(gray)
        get_landmarks(clahe_image)
        if data['landmarks_vectorised'] == "error":
            print("no face detected on this one")
        else:
            prediction_data.append(data['landmarks_vectorised'])
            # prediction_labels.append(emotions.index(emotion))
    return  prediction_data

accur_lin = []


print("Making sets " ) #Make sets by random sampling 80/20%
prediction_data = make_sets()
npar_train = np.load(sys.argv[2]) #Turn the training set into a numpy array for the classifier
npar_trainlabs = np.load(sys.argv[3])
print("training SVM linear " ) #train SVM
clf1 = SVC(kernel='linear', probability=True, C=10)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel

npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
npar_trainlabs = np.array(training_labels)
npar_pred = np.array(prediction_data)
npar_predlabs = np.array(prediction_labels)


Classifier1 = clf1.fit(npar_train, training_labels)


# predictions
PredicoesClf1 = clf1.predict(npar_pred)
# probabilities
ProbabilidadeClf1 = clf1.predict_proba(npar_pred)




OddsCLf1 = pd.DataFrame(data=ProbabilidadeClf1,index=range(len(ProbabilidadeClf1)),columns=emotions)
OddsCLf1 ['Classe'] = PredicoesClf1
OddsCLf1 ['Names'] = [ item for elem in prediction_names for item in elem]
OddsCLf1 ['Prediction'] = prediction_labels
OddsCLf1 ['Errors'] =  OddsCLf1 ['Classe'] - OddsCLf1 ['Prediction']
OddsCLf1.to_csv('SVC_Probabilidades.csv', sep=';')

Errors = OddsCLf1[OddsCLf1.Errors != 0]
Errors.to_csv('Errors.csv',mode='a', header=False, sep=';')

        
