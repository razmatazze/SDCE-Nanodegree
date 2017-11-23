
# Vehicle Detection

## Udacity Self-Driving-Car-Engineer Nanodegree 2017, April


### by Christoph Reiners

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


[image1]: ./images/explore.png "fig1: vehicle images"
[image2]: ./images/explore2.png "fig2: non-vehicle images"
[image3]: ./images/hog.png "fig3: HOG car"
[image4]: ./images/hog2.png "fig4: HOG non-car"
[image5]: ./images/heat.png "fig5: heat map"
[image6]: ./images/cars.png "fig6: thresholded heat map and winows drawn on image"

---
## Dataset and Feature Extraction

The project repository provides an image dataset of vehicle and non-vehicle images. Their features will be extracted before beeing trained on the classifier.

### Exploration of the Dataset

The provided dataset contains in total 8792 images of cars and 8968 images that feature non-vehicles. The resolution of the images is 64x64 and they are in RGB-colourspace. The images are taken from both the GTI and KITTI datasets. Fig1 and Fig2 show example images of each class. ![] [image1] ![] [image2]

### Feature Extraction

For a classifier to work reliable, it needs good input data. Good is considered in this case, as the goal is to classify images, to extract features from the images. Without a good feature extraction, e.g. the features are not generalized enough, the classifier will most likely fail on anything but the training set. The suggested method from the lesson is to use Histogram of Oriented Gradient (HOG). The extracted HOG features are shown for two examples for cars and non-vehicle objects both in Fig.3 and Fig4. ![] [image3] ![] [image4]

---
## Classifier and Training

The next step is to train a classifier on the images with these features extracted. Therefore the dataset is split into a training and test set. The trained classifier will then be applied to real images and a video to ultimately validate the precidtion's accuracy.

The linear support vector classification is implemented, as suggested and presented in the lessons.

### Training
Due to the high computation duration, the validation on the project video was done only every few changes if the increase in accuracy was satisfying. Also the project video was cut to save computation time during the training. The following table shows the results of the trial and error parametrization. The second column contains the parametrization of the HOG parameters: orientations pixels per cell and cells per block.


Test | C-Space | (orien,pix,cell) | len(FeatureVector) | predict duration for 10 images | accuracy | comment
--- | --- | --- | --- | -- | --- | --- |
1 | R (of RGB) | (6,6,2) | 1944 | 0.006s | 0.87 |
2 | G (of RGB) | (6,6,2) | 1944 | 0.005s | 0.9167
3 | RGB | (6,6,2) | 5832 | 0.006s | 0.9333 |
4 | HSV | (6,6,2) | 5832 | 0.007s | 0.9683 |
5 | LUV | (6,6,2) | 5832 | 0.006s | 0.965 |
6 | HLS | (6,6,2) | 5832 | 0.005s | 0.975 |
7 | YUV | (6,6,2) | 5832 | 0.006s | 0.9783 | YUV was the best, check other channels
8 | YCrCb | (6,6,2) | 5832 | 0.006s | 0.97 |
9 | Y (of YUV) | (6,6,2) | 1944 | 0.006s | 0.9133 |
10 | U (of YUV) | (6,6,2) | 1944 | 0.006s | 0.9167 |
11 | V (of YUV) | (6,6,2) | 1944 | 0.006s | 0.87 |
12 | YUV | (8,6,2) | 7776 | 0.007s | 0.9733 |
13 | YUV | (10,6,2) | 9720 | 0.009s | 0.98 |
14 | YUV | (7,8,2) | 9720 | 0.006s | 0.9717 |
15 | YUV | (7,10,2) | 2100 | 0.006s | 0.98 |
16 | YUV | (7,10,4) | 3024 | 0.006s | 0.965 |
17 | YUV | (7,10,6) | 756 | 0.005s | 0.975 |
18 | YCrCb | (7,6,6) | 37044 | 0.01s | 0.9617 |
19 | YCrCb | (9,8,2) | 5292 | 0.006s | 0.98 |
20 | YCrCb | (8,8,2) | 4704 | 0.006s | 0.98 |
21 | YCrCb | (12,18,2) | 576 | 0.0156s | 0.9814 |
22 | YCrCb | (10,12,4) | 1920 | 0.4s | 0.9814 |
23 | YCrCb | (12,10,2) | 3600 | 0.0?s | 0.9891 |

---
## Vehicle Detection

To detect vehicles within an image, parts of the image need to be checked and fed into the classifier. The selection of what parts are fed into the classifier to do its' work is implemented by the sliding windows method.

### Sliding Windows

The first step is to define rectangles. From their content features will be extracted. The feature extractions will then be fed into the trained classifier, to classify the content to either cars or non-vehicle objects. It is suggested to use rectangles of different sizes. Looking at the training set this makes sense, as mostly all vehicle images have the same proportions. So the rectangles should fit cars of different sizes, and thus distances, into their borders to create these proportions the classifier was trained on.

The second step is to define, where the rectangles should be generated and evaluated. Good results can be achieved if the single rectangles will overlap each other but also will increase computing time.




```python
    windows_l = slide_window(image, x_start_stop=[int(1280*0.45), None], y_start_stop=[int(720*0.5), None], 
                        xy_window=(130, 130), xy_overlap=(0.55, 0.75))
    windows_m = slide_window(image, x_start_stop=[int(1280*0.45), None], y_start_stop=[int(720*0.5), int(720*0.95)], 
                        xy_window=(95, 95), xy_overlap=(0.55, 0.75))
    windows_s = slide_window(image, x_start_stop=[int(1280*0.45), int(1280*0.85)], y_start_stop=[int(720*0.45), int(720*0.6)], 
                        xy_window=(30, 30), xy_overlap=(0.5, 0.75))
    random.shuffle(windows_s)
```

As the detection should run in realtime, which is not fulfilled, but may be optimised to that, it may be a good option to break the window searching at some point that will exceed the realtime-capability. To still be able to check critical parts of the image it should be only possible to break the search of the small windows, as they detect farther away objects anyway. The shuffling is applied on the small windows, as if they will be interrupted, on the next frame other parts will be checked with the small windows, instead of beginning again in one corner and may be interrupted again without getting the unknown/unseen area checked. It is mostly a placeholder and an annotation.


### Heat Map

As the classifier works on every window, for every window a detection may happen. Even a good working classifier will produce errors, either so called false positives, or no detections. An Accuracy of even 99,5% will still generate an error at least every 3 frames, depending on the size of the windows, their search area and their overlap, as these are the parameters that define how often the classification is done.

To filter fals positives out it is suggested to implement a heat map. Each detected window will add a value of one to each pixel within the window. speaking exemplary, overlapping detections will generate therefore values >1 whereas single detections remain at value 1. Formulating thresholds for the heat map for each image that was analyzed will filter out false positives, at least if they are weakly recognized by the classifier. Fig.5 and Fig.6 show a heatmap and its' corresponding image with drawn windows around the thresholded "heat-zones". ![] [image5] ![] [image6]

### Improving Heat Map

Instead of just use the threshold of heatmaps, the heatmaps of past frames are stored in a deque class.
    heatmap     = collections.deque(maxlen=25)
Before applying the threshold, the sum is computed. This serves to punish only single appearing but still confident false positives. If there are 4-5 detections as car on e.g. a traffic sign for about 2 frames, will result in a heatmap-value sum of about 8-10. A confident detection of a car over several frames is truly confident and reliable and will result in a sum of >10. So single time confident misdetections can be filtered out.

Furthermore, to prevent from building up heatmapvalue-sums, if the detection is unconfident and classifies too much as car in a specific point, the average heatmap values are computed and thresholded. So even if the at some point false positives exceed the sum-threshold by having themselves accumulated over time, the average-threshold will filter them out, if the confidence from one frame to another, and therefore the heatmap values from one frame to another, are rather low.


```python
    hot_windows = search_windows(image, ...)

    heatmap_cur = np.zeros_like(image[:,:,0]).astype(np.float)
    heatmap_cur = add_heat(heatmap_cur, hot_windows)

    heatmap.append(heatmap_cur)
    #SUM Threshold
    heatmap_s = sum(heatmap)
    heatmap_s=apply_threshold(heatmap_s,18)
    heatmap_sum.append(heatmap_s)
    
    #Average Threshold
    heatmap_avg = np.mean(heatmap_sum, axis=0)
    heatmap_avg=apply_threshold(heatmap_avg,14)

    labels = label(heatmap_avg)
    labels=draw_labeled_bboxes(image, labels)
```

---
## Results and Discusion

The following video shows the results for the settings from the P5_Vehicle_Detection.ipynb.


```python
%%HTML
<video width="900" height="500" controls>
  <source src="project_video.mp4" type="video/mp4">
</video>
```


<video width="900" height="500" controls>
  <source src="project_video.mp4" type="video/mp4">
</video>


The results are not that satisfying. On the other hand, just a feature extraction of the HOG-features was implemented. Especially the short loss of detection may be critical. This could be resolved with a logic that checks, why the detection of a in the past confident detection is lost. E.g. by checking with higher overlapping in this area and untill finding the car again keeping the last known detections. But to implement this, I have to understand the label tool a bit more. Right now I dont quite get how the adressing works.

What I also dont quite like, is that the windows are rather wide and jittering around the cars. This is due to the summation of the heatmap. Before implementing it, the windows were more tight on the cars, but also many many fals positives.
This could be overcome to just use the sum and average thresholded heat maps just as a filter layer, and apply the drawing/deriving of the area just on the base heat map but only in areas where detection is "allowed". But this was not yet tested, on the last day of term :/
