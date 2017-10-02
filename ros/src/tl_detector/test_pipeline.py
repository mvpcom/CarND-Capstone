import glob
#import os
#import cv2
#import math
#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
#import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
#from sklearn.cross_validation import train_test_split  # for sklearn > 0.17 use sklearn.model_selection instead
from sklearn.model_selection import train_test_split 
from helper_functions import *
import time

# IMPORT TRAFFIC LIGHT DATASET
redlights = []
nonredlights = []
# Read in redlight samples
images_red = glob.glob("dataset/sim/red/*.png")
for image in images_red:
    redlights.append(image)
# Read in non-redlight samples
images_notred = glob.glob("dataset/sim/notred/*.png")
for image in images_notred:
    nonredlights.append(image)
print("dataset imported... extracting features and start training... ")


# SET PARAMETERS FOR FEATURE EXTRACTION
### TODO: Tweak these parameters for optimization and visualization.
color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations, usually between 6 and 12
pix_per_cell = 14  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 1  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 196  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

y_start_stop = [400, 700]  # Min and max in y to search in slide_window()
x_start_stop = [380, 1280]  # Min and max in x to search in slide_window()
xy_window = (80, 80)  # size of sliding windows
xy_overlap = (0.55, 0.55)  # overlap fraction of sliding windows
line_color = (0, 255, 102)  # i.e. turquoise : (0, 255, 102), pink: (253, 43, 255)
line_color_2 = (147, 164, 158)  # i.e. light gray: (147, 164, 158), light cyan: (127, 222, 187)
line_thickness = 3  # lines of bounding boxes marking detected vehicles
heat_thresh = 0.75  # threshold for heatmap filter process

# Extract features using above defined parameters
redlights_features = extract_features(redlights, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
nonredlights_features = extract_features(nonredlights, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)


# PREPARE DATASETS FOR TRAINING
X = np.vstack((redlights_features, nonredlights_features)).astype(np.float64)

print("len(redlights): ", len(redlights)) 
print("len(nonredlights): ", len(nonredlights)) 
print("len(redlights_features): ", len(redlights_features))
print("len(nonredlights_features): ", len(nonredlights_features))
print("X.shape: ", X.shape)

  
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(redlights_features)), np.zeros(len(nonredlights_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
print('X_train length total:', len(X_train))


# CREATE AND TRAIN CLASSIFIER
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()


# TESTING PIPELINE
file_list = []
test_images = glob.glob("test_images/*.png")
for img in test_images:
    file_list.append(img)
print("Processing images from folder test_images.")
count = 1
for file_name in file_list:
    image = cv2.imread(file_name)
    # Search for traffic light
    prediction = search_windows(image, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)
    print("prediction: %f" %prediction)
    filename_new = str(count) + "predicted_" + str(prediction) + ".png"
    cv2.imwrite("test_images/results/" + filename_new, image)
    count += 1
print("All test images processed and exported.")
