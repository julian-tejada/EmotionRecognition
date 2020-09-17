# Emotion Recognition 
It is the repository for the code presented on the article **Building and validation of a set of facial expression images to detect emotions: a transcultural study** by Julian Tejada, Raquel Meister Ko. Freitag ,Bruno Felipe Marques Pinheiro, Paloma Batista Cardoso, Victor Rene Andrade Souza, Lucas Santos Silva.
The code is divided in three scripts, which represent the three necessary steps to process a video:
1. Extract the frames from the video
2. Recognize faces o each frame and cut them into a png B&W image
3. Classify the faces B&W image using some of the images dataset. 

## To extract the frames
`python Script_CutVideos.py <name of the video file (mp4)> <Images folder path> <start frame> <stop frame>`

## To cut the faces of the frames
`python ScriptExtractingFaces.py <Source images path> <Destination path (Faces B&W images)>` 

## To classify 
`python Script_EmotionClassification <Source images path> <dataset file (npy)>`

The second and third scripts need the landkmarks descriptions find in the next files
1. From the dlib project the  [Shape predictor 69 face landmarks](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
2. From OpenCV project the [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml), [haarcascade_frontalface_alt.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml), [haarcascade_frontalface_alt2.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt2.xml), [haarcascade_frontalface_alt2.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt2.xml)

# Datasets
Two different datasets of landmarks extracted from images of face expressions (six basic emotion plus mockery) from a transcultural sample of participants (Brazilian and Colombian subjects). The dataset are in two different files, one whit the landmarks (Train_Brazil.npy and  Train_Colombia.npy) and other with the class (Train_Brazil_labels.npy and  Train_Colombia_labels.npy).

