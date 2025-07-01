#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import orbslam3
from ament_index_python.packages import get_package_share_directory

class ORBTrackerNode(Node):
    def __init__(self):
        #self.get_logger().info('Initializing ORBTrackerNode...')
        super().__init__('orb_tracker')
        # Paths: use package share directory
        vocab_path = "/home/pratham/Documents/ORB_SLAM3/Vocabulary/ORBvoc.txt"
        config_path = "/home/pratham/Documents/auv_ws/src/simulation/config/config_example.yaml"
        # Initialize ORB-SLAM3 in monocular mode
        self.slam = orbslam3.system(vocab_path, config_path, orbslam3.Sensor.MONOCULAR)
        if hasattr(self.slam, 'Initialize'):
            self.slam.Initialize()
        elif hasattr(self.slam, 'initialize'):
            self.slam.initialize()
        self.get_logger().info('ORB-SLAM3 initialized')

        # CvBridge
        self.bridge = CvBridge()
        # Subscribe to camera topic: /stereo_left/image
        self.sub = self.create_subscription(
            Image, '/stereo_left', self.image_callback, 1)
        # Motion detection state
        self.prev_gray = None
        self.orb_detector = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.trajectory = []
        # Create OpenCV window
        cv2.namedWindow('ORB-SLAM3 Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ORB-SLAM3 Tracking', 1280, 720)

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV BGR image
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CVBridge error: {e}')
            return
        # Timestamp in seconds
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # SLAM tracking
        try:
            if hasattr(self.slam, 'TrackMonocular'):
                cam_pose = self.slam.TrackMonocular(frame, ts)
            elif hasattr(self.slam, 'track_monocular'):
                cam_pose = self.slam.track_monocular(frame, ts)
            else:
                cam_pose = None
        except Exception as e:
            self.get_logger().warn(f'ORB-SLAM3 tracking error: {e}')
            cam_pose = None

        # Motion detection via ORB matching
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        box = None
        if self.prev_gray is not None:
            kp1, des1 = self.orb_detector.detectAndCompute(self.prev_gray, None)
            kp2, des2 = self.orb_detector.detectAndCompute(gray, None)
            if des1 is not None and des2 is not None:
                raw_matches = self.bf.knnMatch(des1, des2, k=2)
                good = []
                for m_n in raw_matches:
                    # only consider pairs of matches
                    if len(m_n) < 2:
                        continue
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                if len(good) >= 10:
                    pts = [kp2[m.trainIdx].pt for m in good]
                    xs = [int(p[0]) for p in pts]
                    ys = [int(p[1]) for p in pts]
                    if xs and ys:
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                        box = (x1, y1, x2, y2)
                        cx, cy = (x1+x2)//2, (y1+y2)//2
                        self.trajectory.append((cx, cy))
        self.prev_gray = gray



        # Visualization
        vis = frame.copy()
        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
        for i in range(1, len(self.trajectory)):
            cv2.line(vis, self.trajectory[i-1], self.trajectory[i], (255,0,0), 2)

        status = 'SLAM OK' if cam_pose is not None else 'SLAM LOST'
        color = (0,255,0) if cam_pose is not None else (0,0,255)
        cv2.putText(vis, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Side-by-side display
        h, w = frame.shape[:2]
        disp_w = 640
        disp_h = int(h * (disp_w / w))
        combo = np.hstack([
            cv2.resize(frame, (disp_w, disp_h)),
            cv2.resize(vis, (disp_w, disp_h))
        ])
        cv2.imshow('ORB-SLAM3 Tracking', combo)
        cv2.waitKey(1)

    def destroy_node(self):
        # Override to shutdown SLAM
        if hasattr(self.slam, 'Shutdown'):
            self.slam.Shutdown()
        elif hasattr(self.slam, 'shutdown'):
            self.slam.shutdown()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ORBTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
