#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:25:54 2020

@author: julan
"""


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pylab as pl
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

# Carrrega e separa os dados das projeções 3D
anger=pd.read_csv('anger.csv', sep=',',header=0)
anger['class'] = 0
anger3D = anger[["X_0","X_1","X_2","X_3","X_4","X_5","X_6","X_7","X_8","X_9","X_10","X_11","X_12","X_13","X_14","X_15","X_16","X_17","X_18","X_19","X_20","X_21","X_22","X_23","X_24","X_25","X_26","X_27","X_28","X_29","X_30","X_31","X_32","X_33","X_34","X_35","X_36","X_37","X_38","X_39","X_40","X_41","X_42","X_43","X_44","X_45","X_46","X_47","X_48","X_49","X_50","X_51","X_52","X_53","X_54","X_55","X_56","X_57","X_58","X_59","X_60","X_61","X_62","X_63","X_64","X_65","X_66","X_67","Y_0","Y_1","Y_2","Y_3","Y_4","Y_5","Y_6","Y_7","Y_8","Y_9","Y_10","Y_11","Y_12","Y_13","Y_14","Y_15","Y_16","Y_17","Y_18","Y_19","Y_20","Y_21","Y_22","Y_23","Y_24","Y_25","Y_26","Y_27","Y_28","Y_29","Y_30","Y_31","Y_32","Y_33","Y_34","Y_35","Y_36","Y_37","Y_38","Y_39","Y_40","Y_41","Y_42","Y_43","Y_44","Y_45","Y_46","Y_47","Y_48","Y_49","Y_50","Y_51","Y_52","Y_53","Y_54","Y_55","Y_56","Y_57","Y_58","Y_59","Y_60","Y_61","Y_62","Y_63","Y_64","Y_65","Y_66","Y_67","Z_0","Z_1","Z_2","Z_3","Z_4","Z_5","Z_6","Z_7","Z_8","Z_9","Z_10","Z_11","Z_12","Z_13","Z_14","Z_15","Z_16","Z_17","Z_18","Z_19","Z_20","Z_21","Z_22","Z_23","Z_24","Z_25","Z_26","Z_27","Z_28","Z_29","Z_30","Z_31","Z_32","Z_33","Z_34","Z_35","Z_36","Z_37","Z_38","Z_39","Z_40","Z_41","Z_42","Z_43","Z_44","Z_45","Z_46","Z_47","Z_48","Z_49","Z_50","Z_51","Z_52","Z_53","Z_54","Z_55","Z_56","Z_57","Z_58","Z_59","Z_60","Z_61","Z_62","Z_63","Z_64","Z_65","Z_66","Z_67"]]
angerClass = anger[["class"]]

deboche=pd.read_csv('Best_co/deboche/deboche.csv', sep=',',header=0)
deboche['class'] = 1
deboche3D = deboche[["X_0","X_1","X_2","X_3","X_4","X_5","X_6","X_7","X_8","X_9","X_10","X_11","X_12","X_13","X_14","X_15","X_16","X_17","X_18","X_19","X_20","X_21","X_22","X_23","X_24","X_25","X_26","X_27","X_28","X_29","X_30","X_31","X_32","X_33","X_34","X_35","X_36","X_37","X_38","X_39","X_40","X_41","X_42","X_43","X_44","X_45","X_46","X_47","X_48","X_49","X_50","X_51","X_52","X_53","X_54","X_55","X_56","X_57","X_58","X_59","X_60","X_61","X_62","X_63","X_64","X_65","X_66","X_67","Y_0","Y_1","Y_2","Y_3","Y_4","Y_5","Y_6","Y_7","Y_8","Y_9","Y_10","Y_11","Y_12","Y_13","Y_14","Y_15","Y_16","Y_17","Y_18","Y_19","Y_20","Y_21","Y_22","Y_23","Y_24","Y_25","Y_26","Y_27","Y_28","Y_29","Y_30","Y_31","Y_32","Y_33","Y_34","Y_35","Y_36","Y_37","Y_38","Y_39","Y_40","Y_41","Y_42","Y_43","Y_44","Y_45","Y_46","Y_47","Y_48","Y_49","Y_50","Y_51","Y_52","Y_53","Y_54","Y_55","Y_56","Y_57","Y_58","Y_59","Y_60","Y_61","Y_62","Y_63","Y_64","Y_65","Y_66","Y_67","Z_0","Z_1","Z_2","Z_3","Z_4","Z_5","Z_6","Z_7","Z_8","Z_9","Z_10","Z_11","Z_12","Z_13","Z_14","Z_15","Z_16","Z_17","Z_18","Z_19","Z_20","Z_21","Z_22","Z_23","Z_24","Z_25","Z_26","Z_27","Z_28","Z_29","Z_30","Z_31","Z_32","Z_33","Z_34","Z_35","Z_36","Z_37","Z_38","Z_39","Z_40","Z_41","Z_42","Z_43","Z_44","Z_45","Z_46","Z_47","Z_48","Z_49","Z_50","Z_51","Z_52","Z_53","Z_54","Z_55","Z_56","Z_57","Z_58","Z_59","Z_60","Z_61","Z_62","Z_63","Z_64","Z_65","Z_66","Z_67"]]
debocheClass = deboche[["class"]]

disgust=pd.read_csv('Best_co/disgust/disgust.csv', sep=',',header=0)
disgust['class'] = 2
disgust3D = disgust[["X_0","X_1","X_2","X_3","X_4","X_5","X_6","X_7","X_8","X_9","X_10","X_11","X_12","X_13","X_14","X_15","X_16","X_17","X_18","X_19","X_20","X_21","X_22","X_23","X_24","X_25","X_26","X_27","X_28","X_29","X_30","X_31","X_32","X_33","X_34","X_35","X_36","X_37","X_38","X_39","X_40","X_41","X_42","X_43","X_44","X_45","X_46","X_47","X_48","X_49","X_50","X_51","X_52","X_53","X_54","X_55","X_56","X_57","X_58","X_59","X_60","X_61","X_62","X_63","X_64","X_65","X_66","X_67","Y_0","Y_1","Y_2","Y_3","Y_4","Y_5","Y_6","Y_7","Y_8","Y_9","Y_10","Y_11","Y_12","Y_13","Y_14","Y_15","Y_16","Y_17","Y_18","Y_19","Y_20","Y_21","Y_22","Y_23","Y_24","Y_25","Y_26","Y_27","Y_28","Y_29","Y_30","Y_31","Y_32","Y_33","Y_34","Y_35","Y_36","Y_37","Y_38","Y_39","Y_40","Y_41","Y_42","Y_43","Y_44","Y_45","Y_46","Y_47","Y_48","Y_49","Y_50","Y_51","Y_52","Y_53","Y_54","Y_55","Y_56","Y_57","Y_58","Y_59","Y_60","Y_61","Y_62","Y_63","Y_64","Y_65","Y_66","Y_67","Z_0","Z_1","Z_2","Z_3","Z_4","Z_5","Z_6","Z_7","Z_8","Z_9","Z_10","Z_11","Z_12","Z_13","Z_14","Z_15","Z_16","Z_17","Z_18","Z_19","Z_20","Z_21","Z_22","Z_23","Z_24","Z_25","Z_26","Z_27","Z_28","Z_29","Z_30","Z_31","Z_32","Z_33","Z_34","Z_35","Z_36","Z_37","Z_38","Z_39","Z_40","Z_41","Z_42","Z_43","Z_44","Z_45","Z_46","Z_47","Z_48","Z_49","Z_50","Z_51","Z_52","Z_53","Z_54","Z_55","Z_56","Z_57","Z_58","Z_59","Z_60","Z_61","Z_62","Z_63","Z_64","Z_65","Z_66","Z_67" ]]
disgustClass = disgust[["class"]]


fear=pd.read_csv('Best_co/fear/fear.csv', sep=',',header=0)
fear['class'] = 3
fear3D = fear[["X_0","X_1","X_2","X_3","X_4","X_5","X_6","X_7","X_8","X_9","X_10","X_11","X_12","X_13","X_14","X_15","X_16","X_17","X_18","X_19","X_20","X_21","X_22","X_23","X_24","X_25","X_26","X_27","X_28","X_29","X_30","X_31","X_32","X_33","X_34","X_35","X_36","X_37","X_38","X_39","X_40","X_41","X_42","X_43","X_44","X_45","X_46","X_47","X_48","X_49","X_50","X_51","X_52","X_53","X_54","X_55","X_56","X_57","X_58","X_59","X_60","X_61","X_62","X_63","X_64","X_65","X_66","X_67","Y_0","Y_1","Y_2","Y_3","Y_4","Y_5","Y_6","Y_7","Y_8","Y_9","Y_10","Y_11","Y_12","Y_13","Y_14","Y_15","Y_16","Y_17","Y_18","Y_19","Y_20","Y_21","Y_22","Y_23","Y_24","Y_25","Y_26","Y_27","Y_28","Y_29","Y_30","Y_31","Y_32","Y_33","Y_34","Y_35","Y_36","Y_37","Y_38","Y_39","Y_40","Y_41","Y_42","Y_43","Y_44","Y_45","Y_46","Y_47","Y_48","Y_49","Y_50","Y_51","Y_52","Y_53","Y_54","Y_55","Y_56","Y_57","Y_58","Y_59","Y_60","Y_61","Y_62","Y_63","Y_64","Y_65","Y_66","Y_67","Z_0","Z_1","Z_2","Z_3","Z_4","Z_5","Z_6","Z_7","Z_8","Z_9","Z_10","Z_11","Z_12","Z_13","Z_14","Z_15","Z_16","Z_17","Z_18","Z_19","Z_20","Z_21","Z_22","Z_23","Z_24","Z_25","Z_26","Z_27","Z_28","Z_29","Z_30","Z_31","Z_32","Z_33","Z_34","Z_35","Z_36","Z_37","Z_38","Z_39","Z_40","Z_41","Z_42","Z_43","Z_44","Z_45","Z_46","Z_47","Z_48","Z_49","Z_50","Z_51","Z_52","Z_53","Z_54","Z_55","Z_56","Z_57","Z_58","Z_59","Z_60","Z_61","Z_62","Z_63","Z_64","Z_65","Z_66","Z_67"]]
fearClass = fear[["class"]]


happiness=pd.read_csv('Best_co/happiness/happiness.csv', sep=',',header=0)
happiness['class'] = 4
happiness3D = happiness[["X_0","X_1","X_2","X_3","X_4","X_5","X_6","X_7","X_8","X_9","X_10","X_11","X_12","X_13","X_14","X_15","X_16","X_17","X_18","X_19","X_20","X_21","X_22","X_23","X_24","X_25","X_26","X_27","X_28","X_29","X_30","X_31","X_32","X_33","X_34","X_35","X_36","X_37","X_38","X_39","X_40","X_41","X_42","X_43","X_44","X_45","X_46","X_47","X_48","X_49","X_50","X_51","X_52","X_53","X_54","X_55","X_56","X_57","X_58","X_59","X_60","X_61","X_62","X_63","X_64","X_65","X_66","X_67","Y_0","Y_1","Y_2","Y_3","Y_4","Y_5","Y_6","Y_7","Y_8","Y_9","Y_10","Y_11","Y_12","Y_13","Y_14","Y_15","Y_16","Y_17","Y_18","Y_19","Y_20","Y_21","Y_22","Y_23","Y_24","Y_25","Y_26","Y_27","Y_28","Y_29","Y_30","Y_31","Y_32","Y_33","Y_34","Y_35","Y_36","Y_37","Y_38","Y_39","Y_40","Y_41","Y_42","Y_43","Y_44","Y_45","Y_46","Y_47","Y_48","Y_49","Y_50","Y_51","Y_52","Y_53","Y_54","Y_55","Y_56","Y_57","Y_58","Y_59","Y_60","Y_61","Y_62","Y_63","Y_64","Y_65","Y_66","Y_67","Z_0","Z_1","Z_2","Z_3","Z_4","Z_5","Z_6","Z_7","Z_8","Z_9","Z_10","Z_11","Z_12","Z_13","Z_14","Z_15","Z_16","Z_17","Z_18","Z_19","Z_20","Z_21","Z_22","Z_23","Z_24","Z_25","Z_26","Z_27","Z_28","Z_29","Z_30","Z_31","Z_32","Z_33","Z_34","Z_35","Z_36","Z_37","Z_38","Z_39","Z_40","Z_41","Z_42","Z_43","Z_44","Z_45","Z_46","Z_47","Z_48","Z_49","Z_50","Z_51","Z_52","Z_53","Z_54","Z_55","Z_56","Z_57","Z_58","Z_59","Z_60","Z_61","Z_62","Z_63","Z_64","Z_65","Z_66","Z_67"]]
happinessClass = happiness[["class"]]


neutral=pd.read_csv('Best_co/neutral/neutral.csv', sep=',',header=0)
neutral['class'] = 5
neutral3D = neutral[["X_0","X_1","X_2","X_3","X_4","X_5","X_6","X_7","X_8","X_9","X_10","X_11","X_12","X_13","X_14","X_15","X_16","X_17","X_18","X_19","X_20","X_21","X_22","X_23","X_24","X_25","X_26","X_27","X_28","X_29","X_30","X_31","X_32","X_33","X_34","X_35","X_36","X_37","X_38","X_39","X_40","X_41","X_42","X_43","X_44","X_45","X_46","X_47","X_48","X_49","X_50","X_51","X_52","X_53","X_54","X_55","X_56","X_57","X_58","X_59","X_60","X_61","X_62","X_63","X_64","X_65","X_66","X_67","Y_0","Y_1","Y_2","Y_3","Y_4","Y_5","Y_6","Y_7","Y_8","Y_9","Y_10","Y_11","Y_12","Y_13","Y_14","Y_15","Y_16","Y_17","Y_18","Y_19","Y_20","Y_21","Y_22","Y_23","Y_24","Y_25","Y_26","Y_27","Y_28","Y_29","Y_30","Y_31","Y_32","Y_33","Y_34","Y_35","Y_36","Y_37","Y_38","Y_39","Y_40","Y_41","Y_42","Y_43","Y_44","Y_45","Y_46","Y_47","Y_48","Y_49","Y_50","Y_51","Y_52","Y_53","Y_54","Y_55","Y_56","Y_57","Y_58","Y_59","Y_60","Y_61","Y_62","Y_63","Y_64","Y_65","Y_66","Y_67","Z_0","Z_1","Z_2","Z_3","Z_4","Z_5","Z_6","Z_7","Z_8","Z_9","Z_10","Z_11","Z_12","Z_13","Z_14","Z_15","Z_16","Z_17","Z_18","Z_19","Z_20","Z_21","Z_22","Z_23","Z_24","Z_25","Z_26","Z_27","Z_28","Z_29","Z_30","Z_31","Z_32","Z_33","Z_34","Z_35","Z_36","Z_37","Z_38","Z_39","Z_40","Z_41","Z_42","Z_43","Z_44","Z_45","Z_46","Z_47","Z_48","Z_49","Z_50","Z_51","Z_52","Z_53","Z_54","Z_55","Z_56","Z_57","Z_58","Z_59","Z_60","Z_61","Z_62","Z_63","Z_64","Z_65","Z_66","Z_67"]]
neutralClass = neutral[["class"]]


sadness=pd.read_csv('Best_co/sadness/sadness.csv', sep=',',header=0)
sadness['class'] = 6
sadness3D = sadness[["X_0","X_1","X_2","X_3","X_4","X_5","X_6","X_7","X_8","X_9","X_10","X_11","X_12","X_13","X_14","X_15","X_16","X_17","X_18","X_19","X_20","X_21","X_22","X_23","X_24","X_25","X_26","X_27","X_28","X_29","X_30","X_31","X_32","X_33","X_34","X_35","X_36","X_37","X_38","X_39","X_40","X_41","X_42","X_43","X_44","X_45","X_46","X_47","X_48","X_49","X_50","X_51","X_52","X_53","X_54","X_55","X_56","X_57","X_58","X_59","X_60","X_61","X_62","X_63","X_64","X_65","X_66","X_67","Y_0","Y_1","Y_2","Y_3","Y_4","Y_5","Y_6","Y_7","Y_8","Y_9","Y_10","Y_11","Y_12","Y_13","Y_14","Y_15","Y_16","Y_17","Y_18","Y_19","Y_20","Y_21","Y_22","Y_23","Y_24","Y_25","Y_26","Y_27","Y_28","Y_29","Y_30","Y_31","Y_32","Y_33","Y_34","Y_35","Y_36","Y_37","Y_38","Y_39","Y_40","Y_41","Y_42","Y_43","Y_44","Y_45","Y_46","Y_47","Y_48","Y_49","Y_50","Y_51","Y_52","Y_53","Y_54","Y_55","Y_56","Y_57","Y_58","Y_59","Y_60","Y_61","Y_62","Y_63","Y_64","Y_65","Y_66","Y_67","Z_0","Z_1","Z_2","Z_3","Z_4","Z_5","Z_6","Z_7","Z_8","Z_9","Z_10","Z_11","Z_12","Z_13","Z_14","Z_15","Z_16","Z_17","Z_18","Z_19","Z_20","Z_21","Z_22","Z_23","Z_24","Z_25","Z_26","Z_27","Z_28","Z_29","Z_30","Z_31","Z_32","Z_33","Z_34","Z_35","Z_36","Z_37","Z_38","Z_39","Z_40","Z_41","Z_42","Z_43","Z_44","Z_45","Z_46","Z_47","Z_48","Z_49","Z_50","Z_51","Z_52","Z_53","Z_54","Z_55","Z_56","Z_57","Z_58","Z_59","Z_60","Z_61","Z_62","Z_63","Z_64","Z_65","Z_66","Z_67"]]
sadnessClass = sadness[["class"]]


surprise=pd.read_csv('Best_co/surprise/surprise.csv', sep=',',header=0)
surprise['class'] = 7
surprise3D = surprise[["X_0","X_1","X_2","X_3","X_4","X_5","X_6","X_7","X_8","X_9","X_10","X_11","X_12","X_13","X_14","X_15","X_16","X_17","X_18","X_19","X_20","X_21","X_22","X_23","X_24","X_25","X_26","X_27","X_28","X_29","X_30","X_31","X_32","X_33","X_34","X_35","X_36","X_37","X_38","X_39","X_40","X_41","X_42","X_43","X_44","X_45","X_46","X_47","X_48","X_49","X_50","X_51","X_52","X_53","X_54","X_55","X_56","X_57","X_58","X_59","X_60","X_61","X_62","X_63","X_64","X_65","X_66","X_67","Y_0","Y_1","Y_2","Y_3","Y_4","Y_5","Y_6","Y_7","Y_8","Y_9","Y_10","Y_11","Y_12","Y_13","Y_14","Y_15","Y_16","Y_17","Y_18","Y_19","Y_20","Y_21","Y_22","Y_23","Y_24","Y_25","Y_26","Y_27","Y_28","Y_29","Y_30","Y_31","Y_32","Y_33","Y_34","Y_35","Y_36","Y_37","Y_38","Y_39","Y_40","Y_41","Y_42","Y_43","Y_44","Y_45","Y_46","Y_47","Y_48","Y_49","Y_50","Y_51","Y_52","Y_53","Y_54","Y_55","Y_56","Y_57","Y_58","Y_59","Y_60","Y_61","Y_62","Y_63","Y_64","Y_65","Y_66","Y_67","Z_0","Z_1","Z_2","Z_3","Z_4","Z_5","Z_6","Z_7","Z_8","Z_9","Z_10","Z_11","Z_12","Z_13","Z_14","Z_15","Z_16","Z_17","Z_18","Z_19","Z_20","Z_21","Z_22","Z_23","Z_24","Z_25","Z_26","Z_27","Z_28","Z_29","Z_30","Z_31","Z_32","Z_33","Z_34","Z_35","Z_36","Z_37","Z_38","Z_39","Z_40","Z_41","Z_42","Z_43","Z_44","Z_45","Z_46","Z_47","Z_48","Z_49","Z_50","Z_51","Z_52","Z_53","Z_54","Z_55","Z_56","Z_57","Z_58","Z_59","Z_60","Z_61","Z_62","Z_63","Z_64","Z_65","Z_66","Z_67"]]
surpriseClass = surprise[["class"]]



Training3D = pd.concat([anger3D,deboche3D, disgust3D,fear3D,happiness3D,neutral3D,sadness3D,surprise3D], axis=0)
labels3D = pd.concat([angerClass, debocheClass, disgustClass, fearClass, happinessClass, neutralClass, sadnessClass, surpriseClass])

# Carrrega e separa os dados das FACS

angerFACS = anger[["AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r","AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r","AU26_r","AU45_r","AU01_c","AU02_c","AU04_c","AU05_c","AU06_c","AU07_c","AU09_c","AU10_c","AU12_c","AU14_c","AU15_c","AU17_c","AU20_c","AU23_c","AU25_c","AU26_c","AU28_c","AU45_c"]]


debocheFACS = deboche[["AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r","AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r","AU26_r","AU45_r","AU01_c","AU02_c","AU04_c","AU05_c","AU06_c","AU07_c","AU09_c","AU10_c","AU12_c","AU14_c","AU15_c","AU17_c","AU20_c","AU23_c","AU25_c","AU26_c","AU28_c","AU45_c"]]



disgustFACS = disgust[["AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r","AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r","AU26_r","AU45_r","AU01_c","AU02_c","AU04_c","AU05_c","AU06_c","AU07_c","AU09_c","AU10_c","AU12_c","AU14_c","AU15_c","AU17_c","AU20_c","AU23_c","AU25_c","AU26_c","AU28_c","AU45_c"]]


fearFACS = fear[["AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r","AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r","AU26_r","AU45_r","AU01_c","AU02_c","AU04_c","AU05_c","AU06_c","AU07_c","AU09_c","AU10_c","AU12_c","AU14_c","AU15_c","AU17_c","AU20_c","AU23_c","AU25_c","AU26_c","AU28_c","AU45_c"]]


happinessFACS = happiness[["AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r","AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r","AU26_r","AU45_r","AU01_c","AU02_c","AU04_c","AU05_c","AU06_c","AU07_c","AU09_c","AU10_c","AU12_c","AU14_c","AU15_c","AU17_c","AU20_c","AU23_c","AU25_c","AU26_c","AU28_c","AU45_c"]]


neutralFACS = neutral[["AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r","AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r","AU26_r","AU45_r","AU01_c","AU02_c","AU04_c","AU05_c","AU06_c","AU07_c","AU09_c","AU10_c","AU12_c","AU14_c","AU15_c","AU17_c","AU20_c","AU23_c","AU25_c","AU26_c","AU28_c","AU45_c"]]


sadnessFACS = sadness[["AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r","AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r","AU26_r","AU45_r","AU01_c","AU02_c","AU04_c","AU05_c","AU06_c","AU07_c","AU09_c","AU10_c","AU12_c","AU14_c","AU15_c","AU17_c","AU20_c","AU23_c","AU25_c","AU26_c","AU28_c","AU45_c"]]


surpriseFACS = surprise[["AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r","AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r","AU20_r","AU23_r","AU25_r","AU26_r","AU45_r","AU01_c","AU02_c","AU04_c","AU05_c","AU06_c","AU07_c","AU09_c","AU10_c","AU12_c","AU14_c","AU15_c","AU17_c","AU20_c","AU23_c","AU25_c","AU26_c","AU28_c","AU45_c"]]


# divide os bancos de cada emoção em 80% e 20%, o primeiro para treino o segundo para teste
anger3D_training = anger3D.sample(int(len(anger3D)*0.8))
angerClass_training = angerClass[0:len(anger3D_training)]


anger3D_test = anger3D.drop(pd.merge( anger3D, anger3D_training,  left_index=True, right_index=True, how='right').index)
angerClass_test = angerClass[0:len(anger3D_test)]

# divide os bancos de cada emoção em 80% e 20%, o primeiro para treino o segundo para teste
deboche3D_training = deboche3D.sample(int(len(deboche3D)*0.8))
debocheClass_training = debocheClass[0:len(deboche3D_training)]


deboche3D_test = deboche3D.drop(pd.merge( deboche3D, deboche3D_training,  left_index=True, right_index=True, how='right').index)
debocheClass_test = debocheClass[0:len(deboche3D_test)]

# divide os bancos de cada emoção em 80% e 20%, o primeiro para treino o segundo para teste
disgust3D_training = disgust3D.sample(int(len(disgust3D)*0.8))
disgustClass_training = disgustClass[0:len(disgust3D_training)]


disgust3D_test = disgust3D.drop(pd.merge( disgust3D, disgust3D_training,  left_index=True, right_index=True, how='right').index)
disgustClass_test = disgustClass[0:len(disgust3D_test)]

# divide os bancos de cada emoção em 80% e 20%, o primeiro para treino o segundo para teste
fear3D_training = fear3D.sample(int(len(fear3D)*0.8))
fearClass_training = fearClass[0:len(fear3D_training)]


fear3D_test = fear3D.drop(pd.merge( fear3D, fear3D_training,  left_index=True, right_index=True, how='right').index)
fearClass_test = fearClass[0:len(fear3D_test)]

# divide os bancos de cada emoção em 80% e 20%, o primeiro para treino o segundo para teste
happiness3D_training = happiness3D.sample(int(len(happiness3D)*0.8))
happinessClass_training = happinessClass[0:len(happiness3D_training)]


happiness3D_test = happiness3D.drop(pd.merge( happiness3D, happiness3D_training,  left_index=True, right_index=True, how='right').index)
happinessClass_test = happinessClass[0:len(happiness3D_test)]

# divide os bancos de cada emoção em 80% e 20%, o primeiro para treino o segundo para teste
neutral3D_training = neutral3D.sample(int(len(neutral3D)*0.8))
neutralClass_training = neutralClass[0:len(neutral3D_training)]


neutral3D_test = neutral3D.drop(pd.merge( neutral3D, neutral3D_training,  left_index=True, right_index=True, how='right').index)
neutralClass_test = neutralClass[0:len(neutral3D_test)]

# divide os bancos de cada emoção em 80% e 20%, o primeiro para treino o segundo para teste
sadness3D_training = sadness3D.sample(int(len(sadness3D)*0.8))
sadnessClass_training = sadnessClass[0:len(sadness3D_training)]


sadness3D_test = sadness3D.drop(pd.merge( sadness3D, sadness3D_training,  left_index=True, right_index=True, how='right').index)
sadnessClass_test = sadnessClass[0:len(sadness3D_test)]

# divide os bancos de cada emoção em 80% e 20%, o primeiro para treino o segundo para teste
surprise3D_training = surprise3D.sample(int(len(surprise3D)*0.8))
surpriseClass_training = surpriseClass[0:len(surprise3D_training)]


surprise3D_test = surprise3D.drop(pd.merge( surprise3D, surprise3D_training,  left_index=True, right_index=True, how='right').index)
surpriseClass_test = surpriseClass[0:len(surprise3D_test)]


# Concatena os bancos para treino
Training3D = pd.concat([anger3D_training,deboche3D_training, disgust3D_training,fear3D_training,happiness3D_training,neutral3D_training,sadness3D_training,surprise3D_training], axis=0)
labels3D_training = pd.concat([angerClass_training, debocheClass_training, disgustClass_training, fearClass_training, happinessClass_training, neutralClass_training, sadnessClass_training, surpriseClass_training])


# Concatena os bancos para teste
Test3D = pd.concat([anger3D_test,deboche3D_test, disgust3D_test,fear3D_test,happiness3D_test,neutral3D_test,sadness3D_test,surprise3D_test], axis=0)
labels3D_test = pd.concat([angerClass_test, debocheClass_test, disgustClass_test, fearClass_test, happinessClass_test, neutralClass_test, sadnessClass_test, surpriseClass_test])








TrainingFACS = pd.concat([sadnessFACS,debocheFACS, disgustFACS,fearFACS,happinessFACS,neutralFACS,sadnessFACS,surpriseFACS], axis=0)





# converte os panda em numpy arrays

npar_train = Training3D.to_numpy()
npar_trainlabs = labels3D_training.to_numpy()

# converte os panda em numpy arrays

npar_predict = Test3D.to_numpy()
npar_predictlabs = labels3D_test.to_numpy()



# prepara o classificador 1 3D
clf1 = SVC(kernel='linear', probability=True, C=10)

Classifier1 = clf1.fit(npar_train, npar_trainlabs.flatten())


PredicoesClf1 = clf1.predict(npar_predict)
# probabilities
ProbabilidadeClf1 = clf1.predict_proba(npar_predict)

cm = confusion_matrix(labels3D_test, PredicoesClf1,normalize='true')

pl.matshow(cm)
pl.title('Confusion matrix of the classifier')
pl.colorbar()
pl.show()
emotions = [ "anger","deboche","disgust", "fear", "happiness", "neutral", "sadness","surprise"] #Emotion list

pl.figure(figsize=(5.5,4))
# sns.heatmap(cm_df, annot=True)
# pl.title('SVM Linear Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
pl.ylabel('True label')
pl.xlabel('Predicted label')
pl.show()

titles_options = [("Confusion matrix, without normalization", None),("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(Classifier1, npar_predict, npar_predictlabs, display_labels=emotions, cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)


##############################################################################
##############################################################################
## Analiza FACS
##############################################################################
##############################################################################


# divide os bancos de cada emoção em 80% e 20%, o primeiro para treino o segundo para teste
angerFACS_training = angerFACS.sample(int(len(angerFACS)*0.8))
angerClass_training = angerClass[0:len(angerFACS_training)]


angerFACS_test = angerFACS.drop(pd.merge( angerFACS, angerFACS_training,  left_index=True, right_index=True, how='right').index)
angerClass_test = angerClass[0:len(angerFACS_test)]

# divide os bancos de cada emoção em 80% e 20%, o primeiro para treino o segundo para teste
debocheFACS_training = debocheFACS.sample(int(len(debocheFACS)*0.8))
debocheClass_training = debocheClass[0:len(debocheFACS_training)]


debocheFACS_test = debocheFACS.drop(pd.merge( debocheFACS, debocheFACS_training,  left_index=True, right_index=True, how='right').index)
debocheClass_test = debocheClass[0:len(debocheFACS_test)]

# divide os bancos de cada emoção em 80% e 20%, o primeiro para treino o segundo para teste
disgustFACS_training = disgustFACS.sample(int(len(disgustFACS)*0.8))
disgustClass_training = disgustClass[0:len(disgustFACS_training)]


disgustFACS_test = disgustFACS.drop(pd.merge( disgustFACS, disgustFACS_training,  left_index=True, right_index=True, how='right').index)
disgustClass_test = disgustClass[0:len(disgustFACS_test)]

# divide os bancos de cada emoção em 80% e 20%, o primeiro para treino o segundo para teste
fearFACS_training = fearFACS.sample(int(len(fearFACS)*0.8))
fearClass_training = fearClass[0:len(fearFACS_training)]


fearFACS_test = fearFACS.drop(pd.merge( fearFACS, fearFACS_training,  left_index=True, right_index=True, how='right').index)
fearClass_test = fearClass[0:len(fearFACS_test)]

# divide os bancos de cada emoção em 80% e 20%, o primeiro para treino o segundo para teste
happinessFACS_training = happinessFACS.sample(int(len(happinessFACS)*0.8))
happinessClass_training = happinessClass[0:len(happinessFACS_training)]


happinessFACS_test = happinessFACS.drop(pd.merge( happinessFACS, happinessFACS_training,  left_index=True, right_index=True, how='right').index)
happinessClass_test = happinessClass[0:len(happinessFACS_test)]

# divide os bancos de cada emoção em 80% e 20%, o primeiro para treino o segundo para teste
neutralFACS_training = neutralFACS.sample(int(len(neutralFACS)*0.8))
neutralClass_training = neutralClass[0:len(neutralFACS_training)]


neutralFACS_test = neutralFACS.drop(pd.merge( neutralFACS, neutralFACS_training,  left_index=True, right_index=True, how='right').index)
neutralClass_test = neutralClass[0:len(neutralFACS_test)]

# divide os bancos de cada emoção em 80% e 20%, o primeiro para treino o segundo para teste
sadnessFACS_training = sadnessFACS.sample(int(len(sadnessFACS)*0.8))
sadnessClass_training = sadnessClass[0:len(sadnessFACS_training)]


sadnessFACS_test = sadnessFACS.drop(pd.merge( sadnessFACS, sadnessFACS_training,  left_index=True, right_index=True, how='right').index)
sadnessClass_test = sadnessClass[0:len(sadnessFACS_test)]

# divide os bancos de cada emoção em 80% e 20%, o primeiro para treino o segundo para teste
surpriseFACS_training = surpriseFACS.sample(int(len(surpriseFACS)*0.8))
surpriseClass_training = surpriseClass[0:len(surpriseFACS_training)]


surpriseFACS_test = surpriseFACS.drop(pd.merge( surpriseFACS, surpriseFACS_training,  left_index=True, right_index=True, how='right').index)
surpriseClass_test = surpriseClass[0:len(surpriseFACS_test)]


# Concatena os bancos para treino
TrainingFACS = pd.concat([angerFACS_training,debocheFACS_training, disgustFACS_training,fearFACS_training,happinessFACS_training,neutralFACS_training,sadnessFACS_training,surpriseFACS_training], axis=0)
labelsFACS_training = pd.concat([angerClass_training, debocheClass_training, disgustClass_training, fearClass_training, happinessClass_training, neutralClass_training, sadnessClass_training, surpriseClass_training])


# Concatena os bancos para teste
TestFACS = pd.concat([angerFACS_test,debocheFACS_test, disgustFACS_test,fearFACS_test,happinessFACS_test,neutralFACS_test,sadnessFACS_test,surpriseFACS_test], axis=0)
labelsFACS_test = pd.concat([angerClass_test, debocheClass_test, disgustClass_test, fearClass_test, happinessClass_test, neutralClass_test, sadnessClass_test, surpriseClass_test])









# converte os panda em numpy arrays

npar_train = TrainingFACS.to_numpy()
npar_trainlabs = labelsFACS_training.to_numpy()

# converte os panda em numpy arrays

npar_predict = TestFACS.to_numpy()
npar_predictlabs = labelsFACS_test.to_numpy()



# prepara o classificador 1 FACS
clf1 = SVC(kernel='linear', probability=True, C=10)

Classifier1 = clf1.fit(npar_train, npar_trainlabs.flatten())


PredicoesClf1 = clf1.predict(npar_predict)
# probabilities
ProbabilidadeClf1 = clf1.predict_proba(npar_predict)

cm = confusion_matrix(labelsFACS_test, PredicoesClf1,normalize='true')

pl.matshow(cm)
pl.title('Confusion matrix of the classifier')
pl.colorbar()
pl.show()


pl.figure(figsize=(5.5,4))
# sns.heatmap(cm_df, annot=True)
# pl.title('SVM Linear Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
pl.ylabel('True label')
pl.xlabel('Predicted label')
# pl.show()

titles_options = [("Confusion matrix, without normalization", None),("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(Classifier1, npar_predict, npar_predictlabs, display_labels=emotions, cmap=plt.cm.Blues,normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

# emotions = [ "anger","deboche","disgust", "fear", "happiness", "neutral", "sadness","surprise"] #Emotion list

# OddsCLf1 = pd.DataFrame(data=ProbabilidadeClf1,index=range(len(ProbabilidadeClf1)),columns=emotions)
# OddsCLf1 ['Classe'] = PredicoesClf1
# OddsCLf1 ['Names'] = np.arange(0,OddsCLf1.shape[0])
# #OddsCLf1 ['Prediction'] = prediction_labels
# #OddsCLf1 ['Errors'] =  OddsCLf1 ['Classe'] - OddsCLf1 ['Prediction']
# OddsCLf1.to_csv('SVC_Probabilidades.csv', sep=';')

# # Teste das FACS
# PredicoesClf2 = clf2.predict(npar_predict2)
# # probabilities
# ProbabilidadeClf2 = clf2.predict_proba(npar_predict2)



# OddsCLf2 = pd.DataFrame(data=ProbabilidadeClf2,index=range(len(ProbabilidadeClf2)),columns=emotions)
# OddsCLf2 ['Classe'] = PredicoesClf2
# OddsCLf2 ['Names'] = np.arange(0,OddsCLf2.shape[0])
# #OddsCLf1 ['Prediction'] = prediction_labels
# #OddsCLf1 ['Errors'] =  OddsCLf1 ['Classe'] - OddsCLf1 ['Prediction']
# OddsCLf2.to_csv('SVC_Probabilidades2.csv', sep=';')
    