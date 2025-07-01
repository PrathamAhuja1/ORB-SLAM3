#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import orbslam3
import time
import math
from scipy.spatial.transform import Rotation as R
import message_filters

class EnhancedORBTrackerNode(Node):
    def __init__(self):
        super().__init__('enhanced_orb_tracker')
        
        # Paths: use package share directory
        vocab_path = "/home/pratham/Documents/ORB_SLAM3/Vocabulary/ORBvoc.txt"
        config_path = "/home/pratham/Documents/auv_ws/src/simulation/config/config_stereo.yaml"
        
        # Initialize ORB-SLAM3 in STEREO mode
        self.slam = orbslam3.system(vocab_path, config_path, orbslam3.Sensor.STEREO)
        if hasattr(self.slam, 'Initialize'):
            self.slam.Initialize()
        elif hasattr(self.slam, 'initialize'):
            self.slam.initialize()
        self.get_logger().info('ORB-SLAM3 initialized in STEREO mode')

        # CvBridge
        self.bridge = CvBridge()
        
        # Subscribe to stereo camera topics using message filters for synchronization
        self.left_sub = message_filters.Subscriber(self, Image, '/stereo_left')
        self.right_sub = message_filters.Subscriber(self, Image, '/stereo_right')
        
        # Synchronize stereo images
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.stereo_callback)
        
        # Publishers for RViz visualization
        self.pose_pub = self.create_publisher(PoseStamped, '/robot_pose', 10)
        self.velocity_pub = self.create_publisher(TwistStamped, '/robot_velocity', 10)
        self.point_cloud_pub = self.create_publisher(PointCloud2, '/feature_cloud', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/tracking_markers', 10)
        self.debug_image_pub = self.create_publisher(Image, '/debug_image', 10)
        self.disparity_pub = self.create_publisher(Image, '/disparity_image', 10)
        
        # Motion detection and tracking state
        self.prev_left_gray = None
        self.prev_right_gray = None
        self.orb_detector = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Stereo matcher for depth estimation
        self.stereo_matcher = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        
        # Camera pose tracking
        self.current_pose = None
        self.previous_pose = None
        self.previous_timestamp = None
        self.current_linear_velocity = 0.0
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        
        # Feature tracking for coordinates display
        self.current_feature_coords = []
        
        # Create OpenCV windows
    #    cv2.namedWindow('Enhanced ORB-SLAM3 Stereo Tracking', cv2.WINDOW_NORMAL)
    #    cv2.resizeWindow('Enhanced ORB-SLAM3 Stereo Tracking', 1600, 900)
        
        self.get_logger().info('Enhanced ORB Tracker Node initialized in STEREO mode')

    def add_label_to_image(self, image, label, position='top_left'):
        """Add a label to an image section"""
        h, w = image.shape[:2]
        
        # Label properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        
        # Get text size
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # Position coordinates
        if position == 'top_left':
            x, y = 10, 30
        elif position == 'top_right':
            x, y = w - text_size[0] - 10, 30
        elif position == 'top_center':
            x, y = (w - text_size[0]) // 2, 30
        else:
            x, y = 10, 30
        
        # Draw background rectangle
        cv2.rectangle(image, (x-5, y-text_size[1]-5), (x+text_size[0]+5, y+5), bg_color, -1)
        
        # Draw text
        cv2.putText(image, label, (x, y), font, font_scale, text_color, thickness)
        
        return image

    def calculate_linear_velocity(self, current_pose, previous_pose, dt):
        """Calculate linear velocity magnitude from pose change"""
        if dt <= 0 or dt > 1.0:
            return self.current_linear_velocity
        
        try:
            current_trans = current_pose[:3, 3]
            prev_trans = previous_pose[:3, 3]
            displacement = current_trans - prev_trans
            velocity_magnitude = np.linalg.norm(displacement) / dt
            
            if velocity_magnitude < 5.0:
                alpha = 0.7
                self.current_linear_velocity = alpha * velocity_magnitude + (1 - alpha) * self.current_linear_velocity
            
            self.get_logger().info(f'Velocity: {self.current_linear_velocity:.4f} m/s, dt: {dt:.4f}s')
            
        except Exception as e:
            self.get_logger().warn(f'Velocity calculation error: {e}')
        
        return self.current_linear_velocity

    def compute_disparity(self, left_gray, right_gray):
        """Compute disparity map from stereo images"""
        try:
            # Compute disparity
            disparity = self.stereo_matcher.compute(left_gray, right_gray)
            
            # Normalize for visualization
            disparity_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            return disparity, disparity_norm
        except Exception as e:
            self.get_logger().warn(f'Disparity computation error: {e}')
            return None, None

    def track_stereo_features(self, left_frame, right_frame, prev_left_frame):
        """Advanced stereo feature tracking with depth information"""
        if prev_left_frame is None:
            return [], []
        
        # Detect features in left frames
        kp1, des1 = self.orb_detector.detectAndCompute(prev_left_frame, None)
        kp2, des2 = self.orb_detector.detectAndCompute(left_frame, None)
        
        if des1 is None or des2 is None:
            return [], []
        
        # Match features between consecutive left frames
        raw_matches = self.bf.knnMatch(des1, des2, k=2)
        good_matches = []
        
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        # Extract matched points
        matched_features = []
        current_coords = []
        
        for match in good_matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            matched_features.append((pt1, pt2))
            current_coords.append((int(pt2[0]), int(pt2[1])))
        
        self.current_feature_coords = current_coords
        return matched_features, current_coords

    def create_stereo_point_cloud(self, features, disparity, frame_shape):
        """Create PointCloud2 with depth information from stereo"""
        point_cloud = PointCloud2()
        point_cloud.header.stamp = self.get_clock().now().to_msg()
        point_cloud.header.frame_id = "camera_link"
        
        if not features or disparity is None:
            return point_cloud
        
        # Camera parameters (adjust based on your stereo setup)
        focal_length = 525.0  # pixels
        baseline = 0.1  # meters (distance between stereo cameras)
        cx, cy = frame_shape[1] / 2, frame_shape[0] / 2
        
        point_cloud.height = 1
        point_cloud.width = len(features)
        point_cloud.is_bigendian = False
        point_cloud.is_dense = True
        
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        point_cloud.fields = fields
        point_cloud.point_step = 16
        point_cloud.row_step = point_cloud.point_step * point_cloud.width
        
        points_data = []
        for i, (pt1, pt2) in enumerate(features):
            try:
                # Get disparity at feature point
                u, v = int(pt2[0]), int(pt2[1])
                if 0 <= u < disparity.shape[1] and 0 <= v < disparity.shape[0]:
                    d = disparity[v, u]
                    
                    if d > 0:  # Valid disparity
                        # Convert to 3D coordinates
                        z = (focal_length * baseline) / (d / 16.0)  # Disparity is typically scaled
                        x = (u - cx) * z / focal_length
                        y = (v - cy) * z / focal_length
                    else:
                        # Fallback to estimated depth
                        z = 2.0
                        x = (u - cx) * z / focal_length
                        y = (v - cy) * z / focal_length
                else:
                    # Out of bounds, use estimated coordinates
                    z = 2.0
                    x = (u - cx) * z / focal_length
                    y = (v - cy) * z / focal_length
                
                # Color based on depth
                depth_color = min(255, max(0, int(255 * (5.0 - z) / 5.0)))
                rgb = (depth_color << 16) | (128 << 8) | (255 - depth_color)
                
                point_data = np.array([x, y, z], dtype=np.float32).tobytes() + \
                           np.array([rgb], dtype=np.uint32).tobytes()
                points_data.append(point_data)
                
            except Exception as e:
                self.get_logger().warn(f'Point cloud generation error: {e}')
                continue
        
        point_cloud.data = b''.join(points_data)
        return point_cloud

    def create_bounding_box_markers(self, bounding_boxes):
        """Create bounding box markers for RViz"""
        marker_array = MarkerArray()
        
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        marker_id = 0
        
        for i, box in enumerate(bounding_boxes):
            x1, y1, x2, y2 = box
            
            marker = Marker()
            marker.header.frame_id = "camera_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            marker.pose.position.x = (center_x - 320) / 320.0 * 2.0
            marker.pose.position.y = (center_y - 240) / 240.0 * 2.0
            marker.pose.position.z = 1.5
            
            marker.pose.orientation.w = 1.0
            
            width = abs(x2 - x1) / 320.0
            height = abs(y2 - y1) / 240.0
            marker.scale.x = width
            marker.scale.y = height
            marker.scale.z = 0.1
            
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.5
            
            marker_array.markers.append(marker)
            marker_id += 1
        
        return marker_array

    def publish_pose_and_velocity(self, pose_matrix, linear_vel_magnitude, timestamp):
        """Publish pose and velocity information"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = "map"
        
        pose_msg.pose.position.x = float(pose_matrix[0, 3])
        pose_msg.pose.position.y = float(pose_matrix[1, 3])
        pose_msg.pose.position.z = float(pose_matrix[2, 3])
        
        rotation_matrix = pose_matrix[:3, :3]
        r = R.from_matrix(rotation_matrix)
        quat = r.as_quat()
        
        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])
        
        self.pose_pub.publish(pose_msg)
        
        vel_msg = TwistStamped()
        vel_msg.header.stamp = timestamp
        vel_msg.header.frame_id = "map"
        vel_msg.twist.linear.x = float(linear_vel_magnitude)
        vel_msg.twist.linear.y = 0.0
        vel_msg.twist.linear.z = 0.0
        vel_msg.twist.angular.x = 0.0
        vel_msg.twist.angular.y = 0.0
        vel_msg.twist.angular.z = 0.0
        
        self.velocity_pub.publish(vel_msg)

    def stereo_callback(self, left_msg, right_msg):
        """Main callback for synchronized stereo images"""
        try:
            left_frame = self.bridge.imgmsg_to_cv2(left_msg, 'bgr8')
            right_frame = self.bridge.imgmsg_to_cv2(right_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CVBridge error: {e}')
            return
        
        # Timestamp in seconds
        ts = left_msg.header.stamp.sec + left_msg.header.stamp.nanosec * 1e-9
        
        # SLAM tracking with stereo images
        cam_pose = None
        try:
            cam_pose = self.slam.process_image_stereo(left_frame, right_frame, ts)
        except Exception as e:
            self.get_logger().warn(f'ORB-SLAM3 stereo tracking error: {e}')
        
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity
        disparity, disparity_norm = self.compute_disparity(left_gray, right_gray)
        
        # Stereo feature tracking
        matched_features, current_coords = self.track_stereo_features(
            left_gray, right_gray, self.prev_left_gray)
        
        # Motion detection and bounding box creation
        bounding_boxes = []
        bounding_box_centers = []
        
        if self.prev_left_gray is not None and matched_features:
            if len(matched_features) >= 10:
                feature_points = [pt2 for pt1, pt2 in matched_features]
                
                if feature_points:
                    xs = [int(p[0]) for p in feature_points]
                    ys = [int(p[1]) for p in feature_points]
                    
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    
                    if (x_max - x_min) > 50 and (y_max - y_min) > 50:
                        bounding_boxes.append((x_min, y_min, x_max, y_max))
                        center_x = (x_min + x_max) // 2
                        center_y = (y_min + y_max) // 2
                        bounding_box_centers.append((center_x, center_y))
        
        # Initialize display values
        linear_velocity_display = self.current_linear_velocity
        roll_display = self.current_roll
        pitch_display = self.current_pitch
        yaw_display = self.current_yaw
        
        # Process camera pose
        if cam_pose is not None and isinstance(cam_pose, np.ndarray):
            try:
                if cam_pose.shape == (4, 4):
                    self.current_pose = cam_pose.copy()
                    
                    if self.previous_pose is not None and self.previous_timestamp is not None:
                        dt = ts - self.previous_timestamp
                        if dt > 0.001:
                            self.current_linear_velocity = self.calculate_linear_velocity(
                                self.current_pose, self.previous_pose, dt)
                    linear_velocity_display = self.current_linear_velocity
                    
                    rotation_matrix = self.current_pose[:3, :3]
                    r = R.from_matrix(rotation_matrix)
                    euler = r.as_euler('xyz', degrees=True)
                    self.current_roll, self.current_pitch, self.current_yaw = euler[0], euler[1], euler[2]
                    roll_display = self.current_roll
                    pitch_display = self.current_pitch
                    yaw_display = self.current_yaw
                    
                    self.publish_pose_and_velocity(
                        self.current_pose, self.current_linear_velocity, left_msg.header.stamp)
                    
                    self.previous_pose = self.current_pose.copy()
                    self.previous_timestamp = ts
                    
                elif hasattr(cam_pose, '__len__') and len(cam_pose) == 16:
                    pose_matrix = np.array(cam_pose).reshape(4, 4)
                    self.current_pose = pose_matrix.copy()
                    
                    if self.previous_pose is not None and self.previous_timestamp is not None:
                        dt = ts - self.previous_timestamp
                        if dt > 0.001:
                            self.current_linear_velocity = self.calculate_linear_velocity(
                                self.current_pose, self.previous_pose, dt)
                    linear_velocity_display = self.current_linear_velocity
                    
                    rotation_matrix = self.current_pose[:3, :3]
                    r = R.from_matrix(rotation_matrix)
                    euler = r.as_euler('xyz', degrees=True)
                    self.current_roll, self.current_pitch, self.current_yaw = euler[0], euler[1], euler[2]
                    roll_display = self.current_roll
                    pitch_display = self.current_pitch
                    yaw_display = self.current_yaw
                    
                    self.publish_pose_and_velocity(
                        self.current_pose, self.current_linear_velocity, left_msg.header.stamp)
                    
                    self.previous_pose = self.current_pose.copy()
                    self.previous_timestamp = ts
                else:
                    self.get_logger().warn(f'Unexpected pose format: {type(cam_pose)}, shape: {getattr(cam_pose, "shape", "no shape")}')
                    
            except Exception as e:
                self.get_logger().error(f'Pose processing error: {e}')
                self.current_pose = np.eye(4)
        else:
            if cam_pose is  None:
                self.get_logger().warn(f'Invalid pose type: {type(cam_pose)}')
        
        # Publish point cloud with depth information
        if matched_features:
            point_cloud = self.create_stereo_point_cloud(matched_features, disparity, left_frame.shape)
            self.point_cloud_pub.publish(point_cloud)
        
        # Publish bounding box markers
        if bounding_boxes:
            markers = self.create_bounding_box_markers(bounding_boxes)
            self.marker_pub.publish(markers)
        
        # Publish disparity image
        if disparity_norm is not None:
            try:
                disparity_msg = self.bridge.cv2_to_imgmsg(disparity_norm, 'mono8')
                disparity_msg.header = left_msg.header
                self.disparity_pub.publish(disparity_msg)
            except Exception as e:
                self.get_logger().warn(f'Disparity image publish error: {e}')
        
        # Visualization - Enhanced for stereo
        vis_left = left_frame.copy()
        
        # Draw bounding boxes and center points
        for i, box in enumerate(bounding_boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(vis_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_left, 'STEREO TRACKED', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if i < len(bounding_box_centers):
                center = bounding_box_centers[i]
                cv2.circle(vis_left, center, 8, (255, 0, 0), -1)
                cv2.circle(vis_left, center, 12, (255, 255, 255), 2)
        
        # Display information on the tracking view
        y_offset = 60  # Start below the label
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        cv2.putText(vis_left, f'Linear Velocity: {linear_velocity_display:.3f} m/s', 
                   (10, y_offset), font, font_scale, text_color, thickness)
        y_offset += 25
        
        cv2.putText(vis_left, f'Roll: {roll_display:.2f}°', 
                   (10, y_offset), font, font_scale, text_color, thickness)
        y_offset += 20
        
        cv2.putText(vis_left, f'Pitch: {pitch_display:.2f}°', 
                   (10, y_offset), font, font_scale, text_color, thickness)
        y_offset += 20
        
        cv2.putText(vis_left, f'Yaw: {yaw_display:.2f}°', 
                   (10, y_offset), font, font_scale, text_color, thickness)
        y_offset += 30
        
        if bounding_box_centers:
            cv2.putText(vis_left, 'Stereo Tracking Centers:', 
                       (10, y_offset), font, font_scale, (255, 0, 0), thickness)
            y_offset += 20
            
            for i, center in enumerate(bounding_box_centers):
                cv2.putText(vis_left, f'Center {i+1}: ({center[0]}, {center[1]})', 
                           (10, y_offset), font, 0.5, (255, 0, 0), 2)
                y_offset += 20
        
        # Create quad display with labels: Left, Right, Processed, Disparity
        h, w = left_frame.shape[:2]
        disp_w = 800
        disp_h = int(h * (disp_w / w))
        
        # Resize images
        left_resized = cv2.resize(left_frame, (disp_w, disp_h))
        right_resized = cv2.resize(right_frame, (disp_w, disp_h))
        vis_resized = cv2.resize(vis_left, (disp_w, disp_h))
        
        if disparity_norm is not None:
            disparity_colored = cv2.applyColorMap(disparity_norm, cv2.COLORMAP_JET)
            disparity_resized = cv2.resize(disparity_colored, (disp_w, disp_h))
        else:
            disparity_resized = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
        
        # Add labels to each quadrant
        left_resized = self.add_label_to_image(left_resized, "LEFT CAMERA", 'top_center')
        right_resized = self.add_label_to_image(right_resized, "RIGHT CAMERA", 'top_center')
        vis_resized = self.add_label_to_image(vis_resized, "TRACKING", 'top_center')
        disparity_resized = self.add_label_to_image(disparity_resized, "DEPTH MAP", 'top_center')
        
        # Create 2x2 grid
        top_row = np.hstack([left_resized, right_resized])
        bottom_row = np.hstack([vis_resized, disparity_resized])
        combo = np.vstack([top_row, bottom_row])

        # Create window with proper sizing
        cv2.namedWindow('Enhanced ORB-SLAM3 Stereo Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Enhanced ORB-SLAM3 Stereo Tracking', combo.shape[1], combo.shape[0])
        
        cv2.imshow('Enhanced ORB-SLAM3 Stereo Tracking', combo)
        cv2.waitKey(1)
        
        # Publish debug image
        try:
            debug_img_msg = self.bridge.cv2_to_imgmsg(vis_left, 'bgr8')
            debug_img_msg.header = left_msg.header
            self.debug_image_pub.publish(debug_img_msg)
        except Exception as e:
            self.get_logger().warn(f'Debug image publish error: {e}')
        
        # Update previous frames
        self.prev_left_gray = left_gray
        self.prev_right_gray = right_gray

    def destroy_node(self):
        if hasattr(self.slam, 'Shutdown'):
            self.slam.Shutdown()
        elif hasattr(self.slam, 'shutdown'):
            self.slam.shutdown()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = EnhancedORBTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()