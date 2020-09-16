# Emotion Recognition 
Here there is the repository for the code presented on the article **Building and validation of a set of facial expression images to detect emotions: a transcultural study** by Julian Tejada, Raquel Meister Ko. Freitag ,Bruno Felipe Marques Pinheiro, Paloma Batista Cardoso, Victor Rene Andrade Souza, Lucas Santos Silva.
The code is divided in three scripts, which represent the three necessary steps to process a video:
1. Extract the frames from the video
2. Recognize faces o each frame and cut them into a png B&W image
3. Classify the faces B&W image using some of the images dataset. 

## To extract the frames
`python Script_CutVideos.py <name of the video file (mp4)> <Images folder path> <start frame> <stop frame>`

## To cut the faces of the frames
`python ScriptExtractingFaces.py <Source images path> <Destination path (Faces B&W images)>` 

## To classify 
