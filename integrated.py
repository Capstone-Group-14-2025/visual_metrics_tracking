import cv2
import mediapipe as mp
import numpy as np
import time

class IntegratedTracker:
    def __init__(self, camera_index=0, min_detection_confidence=0.3, reference_distance=0.5):
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )
        
        # Initialize SIFT detector and matcher
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Initialize CSRT tracker
        self.tracker = cv2.TrackerCSRT_create()
        self.tracking_initialized = False
        
        # Initialize distance calculator
        self.reference_distance = reference_distance
        self.reference_height = None
        
        # Store tracking data
        self.object_keypoints = None
        self.object_descriptors = None
        self.background_keypoints = None
        self.background_descriptors = None
        self.current_bbox = None
        
    def calibrate_reference(self):
        """Calibrate the reference height for distance estimation"""
        print("Stand at the reference distance and press 'c' to calibrate.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Calculate shoulder and hip positions
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
                
                # Calculate vertical height
                shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * frame.shape[0]
                hip_y = (left_hip.y + right_hip.y) / 2 * frame.shape[0]
                vertical_height = abs(shoulder_y - hip_y)
                
                # Draw calibration line
                cv2.line(frame,
                    (int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1]), int(shoulder_y)),
                    (int((left_hip.x + right_hip.x) / 2 * frame.shape[1]), int(hip_y)),
                    (0, 255, 0), 2)
                
            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                self.reference_height = vertical_height
                print(f"Reference height set: {vertical_height:.2f} pixels")
                break
            elif key == ord('q'):
                break
                
        cv2.destroyWindow("Calibration")
    
    def initialize_tracking(self, frame, bbox):
        """Initialize both CSRT and SIFT tracking"""
        x, y, w, h = bbox
        self.current_bbox = bbox
        
        # Initialize CSRT
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)
        
        # Initialize SIFT
        keypoints, descriptors = self.sift.detectAndCompute(frame, None)
        self.object_keypoints = []
        self.object_descriptors = []
        self.background_keypoints = []
        self.background_descriptors = []
        
        for kp, desc in zip(keypoints, descriptors):
            if x <= kp.pt[0] <= x+w and y <= kp.pt[1] <= y+h:
                self.object_keypoints.append(kp)
                self.object_descriptors.append(desc)
            else:
                self.background_keypoints.append(kp)
                self.background_descriptors.append(desc)
        
        self.object_descriptors = np.array(self.object_descriptors)
        self.background_descriptors = np.array(self.background_descriptors)
        self.tracking_initialized = True
        
    def estimate_distance(self, vertical_height):
        """Estimate distance based on the reference height"""
        if self.reference_height is not None and vertical_height > 0:
            return round(self.reference_distance * (self.reference_height / vertical_height), 2)
        return None
        
    def track_frame(self, frame):
        """Process a single frame with both tracking systems"""
        if not self.tracking_initialized:
            return frame, None, None, None
            
        # 1. Update CSRT tracker
        success, bbox = self.tracker.update(frame)
        
        # 2. Process pose detection
        pose_results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        distance = None
        angle_offset = None
        
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            # Calculate shoulder and hip positions for distance
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            # Calculate vertical height and distance
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * frame.shape[0]
            hip_y = (left_hip.y + right_hip.y) / 2 * frame.shape[0]
            vertical_height = abs(shoulder_y - hip_y)
            distance = self.estimate_distance(vertical_height)
            
            # Calculate angle offset
            user_center_x = (left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1]
            horizontal_offset = user_center_x - (frame.shape[1] / 2)
            normalized_offset = horizontal_offset / (frame.shape[1] / 2)
            angle_offset = max(min(normalized_offset * 90, 90), -90)
        
        # 3. If CSRT fails, try SIFT
        if not success:
            print("CSRT failed, using SIFT backup")
            keypoints, descriptors = self.sift.detectAndCompute(frame, None)
            if len(keypoints) > 0 and self.object_descriptors is not None:
                object_matches = self.flann.knnMatch(descriptors, self.object_descriptors, k=2)
                
                good_points = []
                for i, kp in enumerate(keypoints):
                    if i < len(object_matches) and len(object_matches[i]) == 2:
                        if object_matches[i][0].distance < 0.7 * object_matches[i][1].distance:
                            good_points.append(kp.pt)
                
                if len(good_points) > 0:
                    points = np.array(good_points)
                    x_min, y_min = np.min(points, axis=0)
                    x_max, y_max = np.max(points, axis=0)
                    bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
                    
                    # Reinitialize CSRT with new bbox
                    self.tracker = cv2.TrackerCSRT_create()
                    self.tracker.init(frame, bbox)
        
        # Draw bounding box and info
        if success or bbox is not None:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if distance:
                cv2.putText(frame, f"Distance: {distance:.2f}m", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if angle_offset:
                cv2.putText(frame, f"Angle: {angle_offset:.1f}°", (x, y-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        return frame, bbox, distance, angle_offset
        
    def run(self):
        """Main tracking loop"""
        # First frame for ROI selection
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to get frame")
            return
            
        # Select ROI
        bbox = cv2.selectROI("Select Target", frame, False)
        cv2.destroyWindow("Select Target")
        
        # Initialize tracking
        self.initialize_tracking(frame, bbox)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame, bbox, distance, angle = self.track_frame(frame)
            
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    tracker = IntegratedTracker(camera_index=0, reference_distance=0.5)
    tracker.calibrate_reference()
    tracker.run()