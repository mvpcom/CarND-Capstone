import glob
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split  # for sklearn > 0.17 use sklearn.model_selection instead
from helper_functions import *
import time
from styx_msgs.msg import TrafficLight


class TLClassifier(object):
    def __init__(self):
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
        print("TLClassifier: dataset imported... extracting features and start training... ")

        # SET PARAMETERS FOR FEATURE EXTRACTION
        ### TODO: Tweak these parameters for optimization and visualization.
        self.color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 9  # HOG orientations, usually between 6 and 12
        self.pix_per_cell = 14  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = 1  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (32, 32)  # Spatial binning dimensions
        self.hist_bins = 196  # Number of histogram bins
        self.spatial_feat = True  # Spatial features on or off
        self.hist_feat = True  # Histogram features on or off
        self.hog_feat = True  # HOG features on or off

        # Extract features using above defined parameters
        redlights_features = extract_features(redlights, color_space=self.color_space,
                                              spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                              orient=self.orient, pix_per_cell=self.pix_per_cell,
                                              cell_per_block=self.cell_per_block,
                                              hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                              hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        nonredlights_features = extract_features(nonredlights, color_space=self.color_space,
                                                 spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                                 orient=self.orient, pix_per_cell=self.pix_per_cell,
                                                 cell_per_block=self.cell_per_block,
                                                 hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                                 hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        print("TLClassifier: prepare datasets for training")
        # PREPARE DATASETS FOR TRAINING
        X = np.vstack((redlights_features, nonredlights_features)).astype(np.float64)
	    # Fit a per-column scaler
	    self.X_scaler = StandardScaler().fit(X)     
	    # Apply the scaler to X
	    scaled_X = self.X_scaler.transform(X)
        # Define the labels vector
        y = np.hstack((np.ones(len(redlights_features)), np.zeros(len(nonredlights_features))))
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        print('TLClassifier: Using ', self.orient, 'orientations', self.pix_per_cell,
              'pixels per cell and', self.cell_per_block, 'cells per block')
        print('TLClassifier: Feature vector length ', len(X_train[0]))
        print('TLClassifier: X_train length total ', len(X_train))

        # CREATE AND TRAIN CLASSIFIER
        # Use a linear SVC
        self.svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        self.svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('TLClassifier: Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t = time.time()


    def get_classification(self, input_image):
        """Determines the color of the traffic light in the image
        Args:
            input_image (cv::Mat): image containing the traffic light
        Returns:
            tl_state (int): ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        # CLASSIFY INPUT IMAGE
        prediction = search_windows(input_image, self.svc, self.X_scaler, color_space=self.color_space,
                                        spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                        orient=self.orient, pix_per_cell=self.pix_per_cell,
                                        cell_per_block=self.cell_per_block,
                                        hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                        hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        tl_state = 4 # TrafficLight.UNKNOWN
        if prediction == 1:
            # Red traffic light predicted
            tl_state = 0 # TrafficLight.RED
            print("=== RED detected ===")
        elif prediction == 0:
            # Non-Red traffic light predicted
            tl_state = 2 # TrafficLight.GREEN
            print("=== NONred detected ===")
        else:
            print("=== UNKNOWN detected ===")

        return tl_state
