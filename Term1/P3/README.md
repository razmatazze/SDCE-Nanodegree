
# Behavorial Cloning

## Udacity Self-Driving-Car-Engineer Nanodegree 2017, April

### by Christoph Reiners

Behavorial Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[image1]: ./images/01.png "fig1: captured images from different camera angles "

[image2]: ./images/02.png "fig2: original left, cropped right "

[image3]: ./images/03.png "fig3: examplary recovery situations "

[image4]: ./images/04.png "fig1: image-classes distribution "

---
## Introduction

In this project a convolutional neural network should be used to control the steering of a car with goal to keep the car on a track without leaving the pavement. The car has a fixed speed, so the CNN just have to predict a steering angle.

### Provided Enviroment 
Udacity provided a basic driving-simulator. It contains two tracks, a record function for training mode and an autonomous mode that serves as interface with the CNN architecture.

#### Training Mode
The user can drive the car with multiple controller inputs. The record function captures images with around 15fps and creates a .csv-file that matches the imagefile with the steering-angle at the time of capturing. Along with the steering angle values for speed, throttle and brake are logged, too, but will be unused in this project. The captured images are from three unique angles, left, center and right, shown in Fig[1]. ![][image1]

#### Autonomous Mode

The simulator waits for inputs coming from the provided drive.py. This file creates a server, that the simulator connects to. The simulator sends its's captured images and actual-values for brake,throttle and steering angle and receives corresponding setpoints which serve as input for the car to drive. Drive.py calculates the values for throttle and brake based up on a setpoint for the desired speed and a simple PI-control. The steering angles are the prediction of the CNN (model.h5) for the images drive.py feeds to it.

---
## Dataset

### Input Dimensions and preprocessing
The output of the record-function of the driving-simulator are RGB .jpg-files with the dimension *160x320x3*. Before they are processed in the CNN the dimension and content can be altered. It was tried to decrease the resolution to decrease memory usage and therefore to save computation time but as the implementations induced errors in several ways which couldn't be resolved without putting too much time into it, this approach was abandoned.

Within the CNN the images are cropped from bottom and from top, though. This is done to prevent the network from learning on unnecessary image features. The bottom is cropped by 15 pixels because the bottom area contains information of the present position which are not necessary. More important is the information about where the car will be if it keeps driving in the direction of its' view. The top area of the image on the other hand contains most likely no information about the track at all as for a plane track it will only features the horizon and far away objects. From the top 65 pixels will be cropped and the actual image dimension that is processed by the CNN is *80x320x3*. Fig[2] shows an image's origional and cropped version. Even a more aggressive cropping could be considered but was not tested. ![][image2]

Of course the images are also normalized before beeing processed by the CNN itself. This is done within the network in the very first layer. that will normalize the value range of each pixel of each colourchannel from *int[0,255]* to *float[-1,1]*, which are good values for a neural network, as the outputs internally are also in this range and datatype.

As said, the preprocessing is done within the network in the first two layers of the architecture:

    model=Sequential()
    model.add(Lambda(lambda images: ((images/127.5)-1), input_shape))
    model.add(Cropping2D(cropping=((crop_top=65,crop_bot=15), (0,0)), input_shape))


### Obtaining the Dataset
Udacity provided a small dataset with images of the first track but this was not used, because it was not checked what data exactly was collected. Instead all the images in the dataset were collected by driving by my own on the track in the simulator using the mouse as steering, as this provides an analogous steering angle.
The optimum for this project is that the car drives perfectly centered on the middle of the track. To achieve this, the dataset which is feeded to the network should represent this. Like on most tracks, long straights are overrepresented and corners make mostly a small part. Therefore approximately only six laps in total were driven completely on the track, smooth cornering and with little to no adjustments in the centerline, and in both directions. More data was collected by driving the corner back and forth smoothly on the centerline. But training a network with this data will not teach the network what to do when getting off of the centerline. So a big part of the recording went to recovery situations. The recording was turned off, and the car was driven in positions where it actually should not be when driving autonomously. The steering angle was adjusted towards the centerline at maximum angle and with turned on recording drove back to the centerline with strong and fast steering. Fig[3] shows examplary positions and situations and the desired driving-line. ![][image3]
In autonomous mode the CNN often had trouble to predict correct steering angles when beeing on the bridge, even though there were many recovery maneuvers recorded. My guess was, that the different texture was underrepresented in the dataset and the recognition did not fit to this. But I figured out, that it was due to my recordings. I either recorded recovery maneuvers from wrong positions to the center lane or drove straight with zero steeringangle accross the bridge. So the network could not learn any correct predictions for actually staying on the centerline. I corrected this by driving a few time back and forth the bridge on the centerline, but this time with continous, relative high frequency but low amplitude swerving around this line.
Also I recorded approximately one lap each focusing on recovery and smooth center line driving on track two. Track two has sharper turns, a middle lane line, evelation changes which cause other tracksurfaces to be seen in the distance and much more difficult lightning conditions. Despite the relative small amount of images on this track this helped significantly to improve the steering-angle prediction and was more robust. This is because the CNN learns to generalize and not to be too specialized on the features of track one but on features both tracks share.

#### Data augmentation
To obtain a larger dataset without to waste time on recording is to double the dataset by mirroring the images and multiply the according steering angles by -1. This simple step is done within the image loading, where one image is loaded but the input-dataset grows by two instead of one image.
                
    image = cv2.imread(imagename)
    steering_angle = float(accordingsteeringangle_from_csvfile)
    images.append(image)
    measurements.append(steering_angle)
    ######mirror data vertically#######
    image=np.fliplr(image)
    steering_angle=-steering_angle
    images.append(image)
    measurements.append(steering_angle)

No other data augmentation was considered. Another suggested method to obtain a larger dataset, was to use the images of the left and right camera positions and to adjust the logged steering angle for these images so that the actual off-center camera position will be the center of the car. This was not implemented as figuring out the correct adjustment and coding this logic would have taken more time than just to record more images.

---
## Neural Network Architecture

### Architecture Design and Training the Model
For the following discussion of the architecture design process it is to be taken into consideration, that the dataset kept growing, especially for recovery-maneuver imaga data. It is therefore possible that the pointed out performance improvements may happenend because of the dataset and not because of changes to the implemented CNN.

The first serious attempt after testing the basic functionality of my code was the Nvidia CNN architecture presented in the lessons leading up to this project.


```python
#CNN Architecture taken from (c) by Nvidia, implemented by (c) Udacity 
model.add(Convolution2D(24,(5,5), strides=(2,2),activation="relu"))
model.add(Convolution2D(36,(5,5), strides=(2,2),activation="relu"))
model.add(Convolution2D(48,(5,5), strides=(2,2),activation="relu"))
model.add(Convolution2D(64,(3,3), activation="relu"))
model.add(Convolution2D(64,(3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
```

- 32058 training images out of  37716 overall images [ 15.0 % for validation set]
- 800 batch-size
---
    Epoch 1/5 loss: 40.2280 - val_loss: 11.4453
    Epoch 2/5 loss: 4.5891 - val_loss: 1.2830
    Epoch 3/5 loss: 0.9326 - val_loss: 0.6150
    Epoch 4/5 loss: 0.3673 - val_loss: 0.2114
    Epoch 5/5 loss: 0.1359 - val_loss: 0.1609
    
No overfitting apparent and still reducing loss. Driving was ok, but the car turns slightly right and gets off the track and keeps driving next to the curbs offroad.

After varying the the batch sizes and maximum epochs I decided to implement dropout layers. But as overfitting seems to be of no issue yet, this did not improve the results much. I did read in some discussions that the adam optimizer may not be that good. So I tried a few optimiziers available in keras and decided on the stochastic gradient descent optimizer. Without much knowledge about how it works I just tried out with non-default settings.


```python
model.add(Convolution2D(24,(5,5), strides=(2,2),activation="relu"))
model.add(Dropout(dropout1))
model.add(Convolution2D(36,(5,5), strides=(2,2),activation="relu"))
model.add(Dropout(dropout2))
model.add(Convolution2D(48,(5,5), strides=(2,2),activation="relu"))
model.add(Dropout(dropout3))
model.add(Convolution2D(64,(3,3), activation="relu"))
model.add(Dropout(dropout4))
model.add(Convolution2D(64,(3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

sgd = optimizers.SGD(lr=learning_rate, decay=decay, momentum=nesterov_momentum, nesterov=True)
model.compile(loss='mse', optimizer=sgd)
```

Dropout-Rates: Layer 1-2: 1 %, Layer 2-3: 2 %, Layer 3-4: 30 %, Layer 4-5: 50 %
33644 training images out of  39582 overall images [ 15.0 % for validation set]
learning=0.008, decay=1e-6, nesterovmomentum=0.4
250 batch-size

    Epoch 1/10 loss: 0.0364 - val_loss: 0.0364
    [...]
    Epoch 9/10 loss: 0.0262 - val_loss: 0.0276
    Epoch 10/10 loss: 0.0258 - val_loss: 0.0274

The loss values the CNN achieved were the best so far. Starting on the straight it performed well, no jittered steering or tendencies to leave the center lane. Heading up to the first corner the car began to steer but really weak. In a result of too weak steering the car drove out of the corner.

After adjusting the parameters to:
33644 training images out of  39582 overall images [ 15.0 % for validation set]
learning=0.015, decay=1e-6, nesterovmomentum=0.7
250 batch-size

    Epoch 1/5 loss: 0.0323 - val_loss: 0.0252
    [...]
    Epoch 4/5 loss: 0.0253 - val_loss: 0.0225
    Epoch 5/5 loss: 0.0247 - val_loss: 0.0220

The car drove really well. But only till it reached the bridge, then it just steered with half power to the right or left and crashing into the wall. Even manually recenter the car or place it near and parallel to the wall did not help the network for better inputs. It just seemed the network did not know what to do on the bridge. After the bridge the driving was almost ok, except the sharp corner with the lake in the background. There the car kept driving straight without steering in the corner and crashing in the lake. So I added a few recordings on the bridge and this specific corner. But this did not help much.

The VGG16 architecture has similar popularity for this kind of task. So I implemented a VGG16 architecture based up on [this code](https://github.com/commaai/research/blob/master/train_steering_model.py). The changes are the dropout layers between the convolutional layers, as I had success with that in the previous project before if there are dropout layers with a small dropout. Also the activation function was changed from ELU to rELU. According to [Martin Heusel](https://www.linkedin.com/pulse/exponential-linear-units-elu-deep-network-learning-martin-heusel)"ELUs have negative values which allows them to push mean unit activations closer to zero" and my feeling tells me, that this may be better towrds the output but not within the model. After completing all projects I will spend a bit of research on the actual concepts and math behind neural networks, as for now I just use them as kind of blackbox tool.


```python
model.add(Convolution2D(16,(8,8), strides=(4, 4), padding="same",activation="relu"))
model.add(Dropout(dropout1))
model.add(Convolution2D(32,(5,5), strides=(2, 2), padding="same",activation="relu"))
model.add(Dropout(dropout2))
model.add(Convolution2D(64,(5,5), strides=(2, 2), padding="same",activation="relu"))
model.add(Flatten())
model.add(Dropout(dropout3))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(dropout4))
model.add(ELU())
model.add(Dense(50))
model.add(Dense(1))
```

37777 training images out of  44444 overall images [ 15.0 % for validation set]
250 batch-size,  0.015 learning rate,  1e-06 decay,  0.7 nesterov-momentum
Dropout-Rates: Layer 1-2: 2 %, Layer 2-3: 3 %, Layer before Dense1: 18 %, Layer before Dense2: 36 %

    Epoch 1/5 loss: 0.0403 - val_loss: 0.0277
    Epoch 2/5 loss: 0.0324 - val_loss: 0.0259
    Epoch 3/5 loss: 0.0305 - val_loss: 0.0247
    Epoch 4/5 loss: 0.0295 - val_loss: 0.0239
    Epoch 5/5 loss: 0.0285 - val_loss: 0.0232

The results were good with a bit of too weak turning in corners. The bridge was crossed bridge perfectly, also with manually taking the car off center or at an angle. But at the sharp corner with the lake in the background also this model drives the car into the lake due to steering too weak.

Further tweaking of the parameters made it a bit better  but the car still ends up in the lake:

37777 training images out of  44444 overall images [ 15.0 % for validation set]
250 batch-size,  0.02 learning rate,  1e-05 decay,  0.85 nesterov-momentum
Dropout-Rates: Layer 1-2: 2 %, Layer 2-3: 3 %, Layer before Dense1: 18 %, Layer before Dense2: 36 %

    Epoch 1/5 loss: 0.0358 - val_loss: 0.0273
    [...]
    Epoch 5/5 loss: 0.0251 - val_loss: 0.0225
    
So the dataset was increased with more recovery data in sharp corners.

47586 training images out of  55984 overall images [ 15.0 % for validation set]
250 batch-size,  0.025 learning rate,  1e-05 decay,  0.85 nesterov-momentum
Dropout-Rates: Layer 1-2: 2 %, Layer 2-3: 3 %, Layer before Dense1: 20 %, Layer before Dense2: 45 %

    Epoch 1/8 loss: 0.0536 - val_loss: 0.0411
    [...]
    Epoch 7/8 loss: 0.0335 - val_loss: 0.0328
    Epoch 8/8 loss: 0.0327 - val_loss: 0.0324

This model already works reliable. The car stays a bit too far outside of a corner but still without touching the lane lines. A few tests with varied parameters did nothing to improve, so I returned to these settings an just let it train for more epochs resulting in:

47586 training images out of  55984 overall images [ 15.0 % for validation set]
250 batch-size,  0.025 learning rate,  1e-05 decay,  0.85 nesterov-momentum
Dropout-Rates: Layer 1-2: 2 %, Layer 2-3: 3 %, Layer before Dense1: 20 %, Layer before Dense2: 45 %

    Epoch 1/15 loss: 0.0589 - val_loss: 0.0402
    Epoch 2/15 loss: 0.0432 - val_loss: 0.0374
    [...]
    Epoch 13/15 loss: 0.0298 - val_loss: 0.0274
    Epoch 14/15 loss: 0.0292 - val_loss: 0.0271
    Epoch 15/15 loss: 0.0289 - val_loss: 0.0266

Car drives very reliable and the issue of driving too far outside has gone. As there is still no overfit apparent, the parametrers could be evaluated and optimized for even better results. But the task is fullfilled so this is the model that is submitted and is stored in the file "model.py".

## Final Model Architecture

The final model architecture from the file "model.py" is a CNN based on a VGG16 architecture based on [this code](https://github.com/commaai/research/blob/master/train_steering_model.py).

<pre>
|  Layer                 |   Input   |   Output  |  Description                                                       |
|------------------------|-----------|-----------|--------------------------------------------------------------------|
|a  Lambda               | 160x320x3 | 160x320x3 | normalized to [-1,+1]                                              |
|b  Cropping2D           | 160x320x3 | 80x320x3  | cropped 65px from top and 15px from bottom                         |
|1  Convolution2D        | 80x320x3  | 80x320x16 | filter 8x8x3x16, strides=(4, 4), padding="same", activation="relu" |
|   Dropout              |           |           | 2% dropout rate                                                    |
|2  Convolution2D        | 80x320x16 | 80x320x32 | filter 5x5x16x32, strides=(2, 2), padding="same", activation="relu"|
|   Dropout              |           |           | 3% dropout rate                                                    |
|3  Convolution2D        | 80x320x32 | 80x320x64 | filter 5x5x32x64, strides=(2, 2), padding="same", activation="relu"|
|4  and Flatten          | 80x320x64 |  1638400  |                                                                    |
|   Dropout + ELU        |           |           | 20% dropout rate                                                   |
|5  Dense                |  1638400  |   512     |                                                                    |
|   Dropout + ELU        |           |           | 45% dropout rate                                                   |
|6  Dense                |    512    |    50     |                                                                    |
|7  Dense                |     50    |     1     |                                                                    |
----------------------------------------------------------------------------------------------------------------------
Loss Function is "mean squared error" , Optimizer is "stochastic gradient descent optimizer" with the parameters
LearningRate = 0.025,  Decay = 1e-05,  Nesterov-Momentum = 0.85 
</pre>

## Results

The network predicts the steering angles robust and reliable. Manually driving the car in bad positions are handeled well to a degree. Putting the car offroad at an angle towards the track sometimes results in driving back on track, but by no means reliable. Also when the car stands too much crossways on the track it is not always possible for the network to turn instead it drives offtrack because not turning hard and long enough.

The second track works pretty well too, except the corners or lanes where another part of the track is seen, then the network predicts wrong steering angles. Given the small training data on this apparent more challenging track it is a good result anyway.

The submitted video of the autonomous drive on track one is seen below.


```python
%%HTML
<video width="320" height="160" controls>
  <source src="video.mp4" type="video/mp4">
</video>
```


<video width="320" height="160" controls>
  <source src="record.mp4" type="video/mp4">
</video>


## Annotations
drive.py is changed in two points only to make driving faster. The PI controller was slightly changed and the setpoint for velocity was raised from 9 to 25, and I am confident that it will work also with 30 but haven't tested yet.


```python

```
