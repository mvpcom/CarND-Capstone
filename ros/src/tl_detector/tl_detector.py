#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, Point
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.lights_closest_wp = []
        self.stop_lines = []
        self.stop_lines_closest_wp = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        self.sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

	rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints
        self.sub2.unregister()

    def traffic_cb(self, msg):
        self.lights = msg.lights
        if (self.waypoints is not None) and (len(self.stop_lines) == 0):
            stops = self.config['stop_line_positions']
            
            for light in self.lights:
                light_pose = light.pose.pose
                self.lights_closest_wp.append(self.get_closest_waypoint(light_pose))
                
            for stop in stops:
                stop_line_pose = Pose()
                stop_line_pose.position = Point()
                stop_line_pose.position.x = stop[0]
                stop_line_pose.position.y = stop[1]
                self.stop_lines.append(stop_line_pose)
                self.stop_lines_closest_wp.append(self.get_closest_waypoint(stop_line_pose))

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1
    
    def get_2D_euc_dist(self, pos1, pos2):
        return math.sqrt((pos1.x-pos2.x)**2 + (pos1.y-pos2.y)**2)
    
    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        distances = []
        pos = pose.position
        for wp in self.waypoints:
            wp_pos = wp.pose.pose.position
            distances.append(self.get_2D_euc_dist(wp_pos, pos))
        return distances.index(min(distances))


    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location
        Args:
            point_in_world (Point): 3D location of a point in the world
        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image
        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)
            #Try new:         
            #base_point = self.listener.transformPoint("/base_link", point_in_world);

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #Use tranform and rotation to calculate 2D position of light in image
	if (trans != None):
		#new: 
	        #base_point = base_point.point
	        #print (base_point)

		#print("rot: ", rot)
		#print("trans: ", trans)
		px = point_in_world.x
		py = point_in_world.y
		pz = point_in_world.z
		xt = trans[0]
		yt = trans[1]
		zt = trans[2]
		#Override focal lengths with data from site for testing
		#fx = 1345.200806
		#fy = 1353.838257
		#Override focal lenghts with manually tweaked values from Udacity forum discussion  
		fx = 2574
		fy = 2744
		#Traffic light's true size
		width_true = 1.0
		height_true = 1.95

		#Convert rotation vector from quaternion to euler:
		euler = tf.transformations.euler_from_quaternion(rot)
		sinyaw = math.sin(euler[2])
		cosyaw = math.cos(euler[2])

		#Rotation followed by translation
		Rnt = (
			px*cosyaw - py*sinyaw + xt,
			px*sinyaw + py*cosyaw + yt,
			pz + zt)
		#print("Rnt: ", Rnt)
		#res = pz + zt
		#print("pz + zt:", res)

		#Pinhole camera model w/o distorion
		#Tweaked:
        	u = int(fx * -Rnt[1]/Rnt[0] + image_width/2-30)
        	v = int(fy * -(Rnt[2]-1.0)/Rnt[0] + image_height+50)
		#Untweaked:
        	#u = int(fx * -Rnt[1]/Rnt[0] + image_width/2)
        	#v = int(fy * -Rnt[2]/Rnt[0] + image_height/2)

		#Get distance tl to car
		distance = self.get_2D_euc_dist(self.pose.pose.position, point_in_world)
		#print("distance: %.2f m" % distance)
		width_apparent = 2*fx*math.atan(width_true/(2*distance))
		height_apparent = 2*fx*math.atan(height_true/(2*distance))
		#print("width_apparent: %.2f " % width_apparent)
		#print("height_apparent: %.2f " % height_apparent)
		#Get points for traffic light's bounding box, top left (tl) and bottom right (br) 
		bbox_tl = (int(u-width_apparent/2), int(v-height_apparent/2))  
		bbox_br = (int(u+width_apparent/2), int(v+height_apparent/2))
	else:
		bbox_tl = (0, 0)
		bbox_br = (0, 0)	
        return (bbox_tl, bbox_br)


    def image_resize(self, scr_img, des_width, des_height):
        """Resizes an image while keeping aspect ratio
        Args: 
            scr_img: image input to resize
            des_width: pixel width of output image 
            des_height: pixel height of output image
        Returns:
            Image: Resized image
        """	
        aspect_ratio_width = des_width/des_height
	    #Have to set manually to 0.5 because divison 30/60 apparentaly results in 0
	    aspect_ratio_width = 0.5
	    aspect_ratio_height = des_height/des_width
	    #print("aspect_ratio_width orig: ", aspect_ratio_width)
	    #print("aspect_ratio_height orig: ", aspect_ratio_height)
        src_height, src_width = scr_img.shape[:2]
        crop_height = int(src_width/aspect_ratio_width)
        height_surplus = (src_height-crop_height)/2
        crop_width = int(src_height/aspect_ratio_height)
        width_surplus = (src_width-crop_width)/2
        #Crop image to keep aspect ratio
        if height_surplus>0:
            crop_img = scr_img[int(height_surplus):(src_height-math.ceil(height_surplus)), 0:src_width]
        elif width_surplus>0:
            crop_img = scr_img[0:src_height, int(width_surplus):(src_width-math.ceil(width_surplus))]
        else: crop_img = scr_img  

        #Resize image
        return cv2.resize(crop_img, (des_width, des_height), 0, 0, interpolation=cv2.INTER_AREA)


    def get_light_state(self, light):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False
	
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
	
	#Convert tl coordinates into pos of tl within img captured by camera
        bbox_tl, bbox_br = self.project_to_image_plane(light.pose.pose.position)

	#TESTING the extraction of traffic light images:
	testing = False
  	if self.save_counter%5 == 0 and testing:
            #Draw point and circle
            #cv2.circle(cv_image, (x,y), 30, (255,255,0), 2)
            #cv2.circle(cv_image, (x,y), 5, (255,255,0), -1)
            #Draw bounding box
            #cv2.rectangle(cv_image, bbox_tl, bbox_br, (255,255,0), 3)
            #Save every 5th frame	
            ##cv2.imwrite('/home/student/imgs/img_{}.jpg'.format(self.save_counter/5), cv_image)
            #print("cv_image exported")
            #Cutting out traffic lights
            tl_image_orig = cv_image[bbox_tl[1]:bbox_br[1], bbox_tl[0]:bbox_br[0]]
            #Resize and save image
            tl_image = self.image_resize(tl_image_orig, 30, 60)
            cv2.imwrite('/home/student/imgs/img_{}.png'.format(self.save_counter/5), tl_image)
            print("tl_image exported")     
        
	#Use light location to zoom in on traffic light in image
	tl_image_orig = cv_image[bbox_tl[1]:bbox_br[1], bbox_tl[0]:bbox_br[0]]
	#Resize image
	tl_image = self.image_resize(tl_image_orig, 30, 60)   
	#Get classification
	tl_state = self.light_classifier.get_classification(tl_image)

	print("status of traffic light: %i" % tl_state)
    return tl_state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        VISIBLE_THRESHOLD = 70
        light = None
        idx_next_light = -1
        
        if self.pose and self.stop_lines:
            car_wp = self.get_closest_waypoint(self.pose.pose)
            bigger_wp = [wp for wp in self.stop_lines_closest_wp 
                              if wp > car_wp]
            if len(bigger_wp) > 0:
                idx_next_light = self.stop_lines_closest_wp.index(min(bigger_wp))
            else:
                idx_next_light = 0
            next_stop_pos = self.stop_lines[idx_next_light].position
            dist_to_next_stop = self.get_2D_euc_dist(self.pose.pose.position, next_stop_pos)
            if dist_to_next_stop <= VISIBLE_THRESHOLD:
                light = self.lights[idx_next_light]
        
        if light:
            state = self.get_light_state(light)
            state = self.lights[idx_next_light].state  # TODO: stop cheating
            return self.stop_lines_closest_wp[idx_next_light], state
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
