# Artifical_Neural_Network_Project
 In this project, I solve the given problem using artificial neural network.

## Problem Definiton

PROBLEM: 

i. In given video, you should scan for the celebrity on the scene (i.e. frame), 

ii. If you found the celebrity, put him/her in a bounding box

iii. Try to register other people and scan also them on the frames.

PART I:

Model Phase 1: 

- USE selected video (download the video on your local disk) focusing on your –at least two- celebrities.

- Construct your Training DataSet (including your celebrity faces images)
  
(The face images should be in variable size. For making easy, you should consider only two cases: Big face and small face)

- Code the model to learn Celebrities faces in two-variable size such as CelebrityA’s small face or big face.

- Write down the code classifying Celebrities’s big faces/small faces and non (at least 5 categories)
  
- Prove that your model classifies and detects the face of celebrity correctly.
  
Just give the cropped image of the celebrity and ask it to your model. (PLEASE Plot THE ACCURACY of TRAINing and TESTing sets at each Epoch.)

- Give the CONFUSION MATRIX of the resultant Model.
  
Location Finding Phase 2:

- Your model must detect your celebrities faces on the frame.
  
- Your model must give the location of the faces of your celebrities.
  
PART II:

Registering Phase 3:

- USE OpenCV’s Face Detector to Find and Register other Faces on the video. Put their face images into “others” dataset (i.e. folder on your disk)
  
- Compare the OTHERS to find out, how many UNIQUE Faces are detected in the video.

## Description Of Pdf File
You can find explanations about the code and the techniques used in the pdf file.
