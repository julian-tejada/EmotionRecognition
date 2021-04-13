# Emotion Recognition 
It is the repository for the code presented on the article **Building and validation of a set of facial expression images to detect emotions: a transcultural study** by Julian Tejada, Raquel Meister Ko. Freitag ,Bruno Felipe Marques Pinheiro, Paloma Batista Cardoso, Victor Rene Andrade Souza, Lucas Santos Silva.

This project intends to provide a classifier algorithm which will accurately perform emotion classification on a video or image.

## Sections:
1. Tools used
2. Datasets
3. Goal of the code
4. About the classifier
5. Setup of the training
6. Results
7. References

# 1. Tools used

We are using GitHub project "OpenFace v. 2.2.0", by GitHub user Tadas Batrusaitis: github.com/TadasBaltrusaitis/OpenFace. In this project, OpenFace is used for processing videos and extracting relevant features which will feed our classifier.

OpenFace outputs a csv file containing several features extracted from the video/image and their values, such as: face landmarks, both in 2 dimensions and 3 dimensions (projected); gaze (estimated direction to which the eyes are pointing); head pose; and so on.

Relevant libraries used in the code (python):
- numpy and pandas for data processing
- sk-learn for the classifier model
- matplotlib for data visualization

# 2. Datasets

There are two different sets of images which are used as our "reference", or "control", or "training sets". The samples of the datasets were taken from brazilian and colombian subjects, and both sets have been previously validated and classified by human experts. Those files are "Train_Brazil.npy" and "Train_Colombia.npy". Alongside the images, there are two label files, namely "Train_Brazil_labels.npy" and "Train_Colombia_labels.npy", which provide the already validated emotion classification for the corresponding images.

The class order in those files are as follows:
0 - anger;
1 - mockery*;
2 - disgust;
3 - fear;
4 - happiness;
5 - neutral;
6 - sadness;
7 - surprise

# 3. Goal of the code

Our goal was to train a classifier algorithm to correctly perform emotion classification. In order to maximize its success rate, we performed several tests so that we could find the best values for the following variables:
1. which feature, or combination of features, provided by OpenFace, would produce the best results in terms of accurate emotion classification; and
2. which values of parameters we should use for the classifier.

The values found for those variables are described in section 6, "Results".

# 4. About the classifier

We use the python machine learning library sk-learn (from sci-kit) to setup the classifier. We are using the Support Vector Classification (SVC) model.

# 5. Setup of the training

In order to train our SVC, we processed the brazilian dataset, "Train_Brazil.npy" using OpenFace. We then obtained 8 csv files containing the values of the features extracted from all the sampled images. Each csv file corresponded to a specific emotion ("anger.csv", "deboche.csv", etc.). In other words, every csv file contained the values of the extracted features of only one particular emotion.

The code for the description below can be found in the "panda_test.py" file.

We converted the csv data into pandas Dataframes in order to allow us to use and manipulate such data. We grouped the csv columns into the following categories:

|        ATTRIBUTE        |           INTERVAL          |
|:-----------------------:|:---------------------------:|
|           gaze          |   [gaze_0_x, gaze_angle_y]  |
|     eye landmarks 2D    | [eye_lmk_x_0, eye_lmk_y_55] |
|     eye landmarks 3D    | [eye_lmk_X_0, eye_lmk_Z_55] |
| head pose (no rotation) |      [pose_Tx, pose_Tz]     |
| head pose (w/ rotation) |      [pose_Rx, pose_Rz]     |
|    face landmarks 2D    |         [x_0, y_67]         |
|    face landmarks 3D    |         [X_0, Z_67]         |
|         poses 1         |       [p_scale, p_ty]       |
|         poses 2         |         [p_0, p_33]         |
|           facs          |       [AU01_r, AU45_c]      |

Additionally, we created a separate Dataframe containing the labels of the emotions. So, if "anger.csv" had, say, 50 entries, we would create 50 entries in the "labels" Dataframe valued 0 (which is the corresponding value to anger in our convention). After those entries, we would have an x number of 1's, where x is the number of entries of the next csv file; and so on.

We splitted the results into a training set and a test set, with a ratio of 80/20, respectively. Such ratio is common practice within machine learning problems. In order to avoid bias in the algorithm, the sampling of which entries go into the sets is random. Thus, the two sets are different every time the code is executed.

We then initialized an SVC classifier and fed it the aforementioned datasets.

Finally, we create a confusion matrix to record and measure the results of the training, and plotted then on a heat map (using matplotlib).

# 6. Results
[requires further text/images to support claims]

After testing the algorithm several times with different values for the variables, we arrived at the following results:

It was found that the single feature which produced the best results was the "FACS" category (columns "AU01_r" to "AU45_c" in the csv files created by OpenFace). Combining the "FACS" category with other attributes only created background noise and negatively affected the success of the classifier.

The best parameter values used in the SVC classifier are as follows:
- kernel: 
- gamma: 
- C:

# 7. References

OpenFace 2.0: Facial Behavior Analysis Toolkit
Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
