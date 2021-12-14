# Environmental-Context-Prediction-for-Lower-Limb-Prostheses

## Project Overview

Types of terrain pose challenge for intelligent prosthetic limbs. Our task is to use accelerometer and gyroscope readings, from sensors attached to the legs of subjects, to predict the type of surface the subject is walking on. There are four types of terrain/activity that are considered: (0) indicates standing or walking on solid ground, (1) indicates going down the stairs,(2) indicates going up the stairs, and (3) indicates walking on grass.
![image](https://user-images.githubusercontent.com/22122136/145943557-c5525ed5-ca07-4a4b-adab-a8ccf268de6d.png)
## MODEL TRAINING AND SELECTION
### A. Model Training
Below is table showing Imbalance in training data.
![image](https://user-images.githubusercontent.com/22122136/145944386-5609f321-4a3e-4261-815b-ff805c6af333.png)
Class 0 is the dominant class while Class 1 is the least present class in the data.
- Step 1: We did an 90-10 split for the training and test sets respectively.
- Step 2: Balancing the training dataset resampling was performed to have equal number, 100000 each, of different labels in training set.
- Step 3: Then we are considering window size of 60 and sliding window of 4, representing every 4 points will determine a single prediction. First prediction will -determine the 15th Y value.
- Step 4: We normalized the data (0, 0.5)

### B. Model Selection
We have implemented 5 different Convolution Neural Network, which have variability on basis of Normalization layer, LSTM layer, Dropout layer. Comparison of best model with other models in shown in Figure 1. (Performance of Adam models)
![image](https://user-images.githubusercontent.com/22122136/145944751-cc0ddf8e-731c-4b8c-b152-057d21d93c15.png)

### C. Best Model
Our 1 D CNN have three convolutional layers, as well as a fully connected linear layer. The first convolutional layer has 6 input channels, representing the X, Y, and Z accelerometer and gyroscope values. The output feature having 12 channels are then fed through normalization layer and the followed by ReLU activation function. Second convolutional layer has 12 input channels and 24 output channels. This is followed by a batch normalization layer, non-linear activation function ReLU and a dropout layer. Third convolutional layer has 24 input channels and 48 output channels, with 5 kernel size and stride of 2. This is followed by a batch normalization, nonlinear activation function ReLU, and a dropout layer. 

The output of the third convolution layer is reshaped to (64, 60, 48) and the passed through 2-layer LSTM model, having size of hidden layer as 128. 

Finally, the output from these layers is flattened and fed into a fully connected layer, where we have three linear transformations, having size 256, 64, 4 respectively. The loss function is calculated with PyTorch’s Cross Entropy Loss. The weights are updated after each iteration using Adam optimizer.

Information regarding best model is in Figure 2.

![image](https://user-images.githubusercontent.com/22122136/145945346-df05c78b-85f1-4116-ad3f-f82b99d18fcd.png)

## III. EVALUATION
After selecting the best model, we retrained this model for 20 epochs using the full training set. Evaluating our model with the full held-out test data, we obtained the below
### Results:
#### Train
Loss: 0.091480
• Accuracy of Standing/Walking on Solid Ground: 99% (99504/100000)
• Accuracy of Up the Stairs: 100% (100000/100000)
• Accuracy of Down the Stairs: 99% (99994/100000)
• Accuracy of Walking on grass: 99%
(99597/100000)
Accuracy (Overall): 99% (399095/400000)
#### Test
Loss: 0.063158
• Accuracy of Standing/Walking on Solid Ground: 98% (24459/24959)
• Accuracy of Up the Stairs: 94% (1439/1531)
• Accuracy of Down the Stairs: 88% (1793/2037)
• Accuracy of Walking on grass: 84% (4176/4971)
Accuracy (Overall): 95% (31867/33498)
Macro F1 score: 0.92
