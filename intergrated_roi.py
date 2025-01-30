import cv2
import mediapipe as mp
import serial
import sys
import time
import numpy as np

# ---------------------------------------------------
#             SIFT-BASED FEATURE TRACKER
# ---------------------------------------------------
class FeatureBasedTracker:
    def __init__(self):
        # Create SIFT and FLANN
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Internals
        self.object_descriptors = None
        self.background_descriptors = None
        self.prev_bbox = None
        self.initialized = False

    def initialize_tracking(self, frame, bbox):
        """
        Initialize the SIFT-based tracker with an ROI specified by bbox.
        bbox = (x, y, w, h)
        """
        self.prev_bbox = bbox

        x, y, w, h = bbox
        # Detect keypoints in entire frame
        keypoints, descriptors = self.sift.detectAndCompute(frame, None)
        if descriptors is None:
            return

        object_keypoints = []
        object_descriptors = []
        background_keypoints = []
        background_descriptors = []

        for kp, desc in zip(keypoints, descriptors):
            if (x <= kp.pt[0] <= x + w) and (y <= kp.pt[1] <= y + h):
                object_keypoints.append(kp)
                object_descriptors.append(desc)
            else:
                background_keypoints.append(kp)
                background_descriptors.append(desc)

        if len(object_descriptors) > 0:
            self.object_descriptors = np.array(object_descriptors, dtype=np.float32)
        else:
            self.object_descriptors = None
        if len(background_descriptors) > 0:
            self.background_descriptors = np.array(background_descriptors, dtype=np.float32)
        else:
            self.background_descriptors = None

        self.initialized = True

    def track_object(self, frame):
        """
        Uses SIFT matching + FLANN to update the bounding box.
        Returns the new_bbox (x, y, w, h) or None if tracking fails.
        """
        if (not self.initialized or
            self.object_descriptors is None or
            self.background_descriptors is None):
            return None

        keypoints, descriptors = self.sift.detectAndCompute(frame, None)
        if descriptors is None or len(keypoints) == 0:
            return None

        # KNN match to object and background
        try:
            object_matches = self.flann.knnMatch(descriptors, self.object_descriptors, k=2)
            background_matches = self.flann.knnMatch(descriptors, self.background_descriptors, k=2)
        except:
            return None

        object_points = []
        for i, kp in enumerate(keypoints):
            if i < len(object_matches) and len(object_matches[i]) == 2:
                # Best match distance for object
                d_o1 = object_matches[i][0].distance
                # For background
                if i < len(background_matches) and len(background_matches[i]) > 0:
                    d_b1 = background_matches[i][0].distance
                else:
                    d_b1 = float('inf')

                if d_b1 == 0:  # avoid division by zero
                    continue

                ratio = d_o1 / d_b1
                if ratio < 0.5:  # accept match
                    object_points.append(kp.pt)

        if len(object_points) == 0:
            return None

        # Use min/max to define bounding box
        points = np.array(object_points)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        new_bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        self.prev_bbox = new_bbox
        return new_bbox


# ---------------------------------------------------
#          CAMERA / MEDIAPIPE POSE
# ---------------------------------------------------
class CameraNode:
    def __init__(self, camera_index=0, width=320, height=240):
        print("Setting Video Capture")
        # Sometimes on Windows you may need cv2.CAP_DSHOW
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        print("Setting CAP_PROP_FRAME_WIDTH")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        print("Setting CAP_PROP_FRAME_HEIGHT")
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()


class PoseEstimationNode:
    def __init__(self, min_detection_confidence=0.3, min_tracking_confidence=0.3):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results


# ---------------------------------------------------
#         DISTANCE / ANGLE CALCULATOR
# ---------------------------------------------------
class DistanceCalculator:
    def __init__(self, reference_distance=0.5):
        self.reference_distance = reference_distance
        self.reference_height = None

    def calibrate(self, vertical_height):
        self.reference_height = vertical_height
        print(f"[CALIBRATION] Reference height set: {vertical_height:.2f} pixels "
              f"for {self.reference_distance} meters.")

    def estimate_distance(self, current_height):
        if self.reference_height is not None and current_height > 0:
            return round(self.reference_distance * (self.reference_height / current_height), 2)
        return None


class MovementController:
    def __init__(self,
                 kv=1.0,
                 kw=1.0,
                 target_distance=0.5,
                 distance_tolerance=0.1,
                 angle_tolerance_deg=25):
        self.target_distance = target_distance
        self.distance_tolerance = distance_tolerance
        self.angle_tolerance_deg = angle_tolerance_deg
        self.kv = kv
        self.kw = kw

    def compute_control(self, distance, angle_offset_deg):
        if distance is None:
            distance_error = 0
        else:
            distance_error = distance - self.target_distance

        angle_error_deg = angle_offset_deg

        linear_vel = self.kv * distance_error
        angular_vel = self.kw * angle_error_deg
        return linear_vel, angular_vel


class SerialOutput:
    def __init__(self, port, baudrate=9600):
        self.ser = serial.Serial(port, baudrate)
        return

    def send_velocities(self, wl, wr):
        msg = f"w_l:{wl:.1f} w_r:{wr:.1f}\n"
        self.ser.write(msg.encode('utf-8'))

    def close(self):
        self.ser.close()


# ---------------------------------------------------
#   DISTANCE-ANGLE TRACKER (MAIN COORDINATION)
# ---------------------------------------------------
class DistanceAngleTracker:
    def __init__(self,
                 camera_index=0,
                 target_distance=0.5,
                 reference_distance=0.5,
                 polling_interval=0.1,
                 port='/dev/ttyUSB0',
                 baudrate=9600,
                 serial_enabled=False,
                 draw_enabled=False,
                 kv=1.0,
                 kw=1.0):
        self.draw_enabled = draw_enabled
        self.serial_enabled = serial_enabled
        self.polling_interval = polling_interval
        self.last_poll_time = time.time()

        # Initialize nodes
        print("Initializing Camera")
        self.camera_node = CameraNode(camera_index=camera_index)
        print("Initializing Pose Node")
        self.pose_node = PoseEstimationNode()
        print("Initializing Distance Calculator")
        self.distance_calculator = DistanceCalculator(reference_distance=reference_distance)
        print("Initializing Controller")
        self.movement_controller = MovementController(kv=kv, kw=kw, target_distance=target_distance)

        if self.serial_enabled:
            print("Initializing Serial Node")
            self.serial_output = SerialOutput(port=port, baudrate=baudrate)
        else:
            self.serial_output = None

        print("Initializing Feature-Based Tracker")
        self.feature_tracker = FeatureBasedTracker()
        self.roi = None  # Store the selected ROI
        self.window_name = "Distance & Angle Tracker"

    def select_roi(self):
        """
        Allows the user to select a region of interest 
        Returns True if ROI was selected, False if cancelled.
        """
        print("Select ROI by dragging the mouse. Press ENTER to confirm or 'c' to cancel.")
        frame = self.camera_node.get_frame()
        if frame is None:
            print("[ERROR] Could not access the camera.")
            return False

        # Use OpenCV's built-in ROI selector
        self.roi = cv2.selectROI("Select Region of Interest", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Region of Interest")

        # Check if ROI was actually selected (width and height > 0)
        if self.roi[2] > 0 and self.roi[3] > 0:
            # Initialize the feature tracker with the selected ROI
            self.feature_tracker.initialize_tracking(frame, self.roi)
            print("ROI selected:", self.roi)
            return True
        else:
            print("ROI selection cancelled")
            return False

    def calibrate_reference(self):
        """
        Allows the user to calibrate the reference height by pressing 'c'.
        Press 'q' to exit early.
        """
        print("Stand at the known distance and press 'c' to calibrate reference height (or 'q' to quit).")

        while True:
            frame = self.camera_node.get_frame()
            if frame is None:
                print("[ERROR] Could not access the camera.")
                break

            results = self.pose_node.process_frame(frame)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Shoulder points
                left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                # Hip points
                left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

                # Calculate the vertical height in pixels
                frame_h, frame_w = frame.shape[:2]
                shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * frame_h
                hip_y = (left_hip.y + right_hip.y) / 2 * frame_h
                vertical_height = abs(shoulder_y - hip_y)

                # Draw calibration line
                shoulder_center_x = int((left_shoulder.x + right_shoulder.x) / 2 * frame_w)
                hip_center_x = int((left_hip.x + right_hip.x) / 2 * frame_w)
                cv2.line(frame,
                         (shoulder_center_x, int(shoulder_y)),
                         (hip_center_x, int(hip_y)),
                         (0, 255, 0), 2)
                cv2.putText(frame, "Calibrate: Press 'c' to set", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Calibrate Reference Distance", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    self.distance_calculator.calibrate(vertical_height)
                    break
                if key == ord('q'):
                    break
            else:
                cv2.imshow("Calibrate Reference Distance", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyWindow("Calibrate Reference Distance")

    def start_tracking(self):
        """
        Main loop that captures frames, estimates distance & angle,
        and maintains tracking with ROI selection.
        Press 'q' to quit, 'r' to reselect ROI, 'c' to recalibrate.
        """
        # First, let user select ROI
        if not self.select_roi():
            print("ROI selection failed or was cancelled. Exiting...")
            return

        while True:
            frame = self.camera_node.get_frame()
            if frame is None:
                print("[ERROR] Could not read frame.")
                break

            # Check user input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if self.serial_output:
                    self.serial_output.send_velocities(0, 0)
                break
            elif key == ord('r'):  # Add ability to reselect ROI
                if self.select_roi():
                    continue
            elif key == ord('c'):
                self.calibrate_reference()

            current_time = time.time()
            if (current_time - self.last_poll_time) >= self.polling_interval:
                self.last_poll_time = current_time

                frame_height, frame_width = frame.shape[:2]
                results = self.pose_node.process_frame(frame)

                # Default values if no Pose is detected
                distance = 0
                linear_vel = 0
                angular_vel = 0
                angle_offset_deg = 0
                user_center_x = frame_width / 2
                user_center_y = frame_height / 2
                pose_detected = False

                if results.pose_landmarks:
                    pose_detected = True
                    landmarks = results.pose_landmarks.landmark
                    # Shoulders
                    left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                    # Hips
                    left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                    right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

                    # Compute the vertical height
                    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * frame_height
                    hip_y = (left_hip.y + right_hip.y) / 2 * frame_height
                    vertical_height = abs(shoulder_y - hip_y)

                    # Estimate user distance
                    distance = self.distance_calculator.estimate_distance(vertical_height)

                    # Compute horizontal offset & angle offset
                    user_center_x = (left_shoulder.x + right_shoulder.x) / 2 * frame_width
                    horizontal_offset = user_center_x - (frame_width / 2)
                    normalized_offset = horizontal_offset / (frame_width / 2)
                    angle_offset_deg = max(min(normalized_offset * 90, 90), -90)

                    # Calculate velocities
                    angle_offset_int = int(angle_offset_deg)
                    if -10 <= angle_offset_int <= 10:
                        angle_offset_int = 0
                    linear_vel, angular_vel = self.movement_controller.compute_control(distance, angle_offset_int)

                    # Convert to wheel velocities or however your system needs
                    radius = 0.5
                    w_hat_l = linear_vel / radius
                    w_hat_r = w_hat_l
                    wl = angular_vel + w_hat_l
                    wr = -angular_vel + w_hat_r

                    # Clip wheel speeds (example)
                    wr = min(max(wr, -0.5), 0.5)
                    wl = min(max(wl, -0.5), 0.5)

                    print("Angle Offset:", angle_offset_int, "Angular Vel:", angular_vel)
                    print("wr:", wr , "wl:", wl)

                    # Send velocities if serial
                    if self.serial_output:
                        self.serial_output.send_velocities(wl, wr)

                    # --- SIFT bounding box update ---
                    # Define a bounding box around shoulders & hips
                    all_x = [
                        left_shoulder.x * frame_width,
                        right_shoulder.x * frame_width,
                        left_hip.x * frame_width,
                        right_hip.x * frame_width
                    ]
                    all_y = [
                        left_shoulder.y * frame_height,
                        right_shoulder.y * frame_height,
                        left_hip.y * frame_height,
                        right_hip.y * frame_height
                    ]
                    x_min, x_max = int(min(all_x)), int(max(all_x))
                    y_min, y_max = int(min(all_y)), int(max(all_y))
                    w = x_max - x_min
                    h = y_max - y_min

                    # Initialize or re-initialize tracker
                    if not self.feature_tracker.initialized:
                        self.feature_tracker.initialize_tracking(frame, (x_min, y_min, w, h))
                    else:
                        # Simple approach: re-init each time
                        self.feature_tracker.initialize_tracking(frame, (x_min, y_min, w, h))

                    self.sift_bbox = (x_min, y_min, w, h)

                else:
                    # --- If Mediapipe fails, fallback to feature-based tracking ---
                    new_bbox = self.feature_tracker.track_object(frame)
                    if new_bbox is not None:
                        self.sift_bbox = new_bbox
                    else:
                        cv2.putText(frame, "Tracking lost!", (50, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                # --- Finally, draw info if enabled ---
                if self.draw_enabled:
                    self._draw_info(
                        frame=frame,
                        distance=distance,
                        linear_vel=linear_vel,
                        angular_vel=angular_vel,
                        angle_offset_deg=angle_offset_deg,
                        user_center_x=user_center_x,
                        user_center_y=user_center_y,
                        bbox=self.sift_bbox,
                        pose_detected=pose_detected
                    )

            cv2.imshow(self.window_name, frame)

        # Cleanup
        self.camera_node.release()
        if self.serial_output:
            self.serial_output.close()
        cv2.destroyAllWindows()

    def _draw_info(self, frame, distance, linear_vel, angular_vel, angle_offset_deg,
                   user_center_x, user_center_y, bbox=None, pose_detected=False):
        """
        Draws:
          - textual info (distance, velocities, etc.)
          - the bounding box from SIFT, if available
          - a "torso line" if pose_detected is True
            (We'll just demonstrate a short line near the user_center for now,
             or you can draw the actual shoulders-hips line in the main code.)
        """
        text_color = (0, 255, 255)
        font_scale = 0.5

        # 1) Text overlays
        cv2.putText(frame, f"Distance: {distance if distance else 0:.2f} m",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
        cv2.putText(frame, f"Linear Vel: {linear_vel:.2f}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
        cv2.putText(frame, f"Angular Vel: {angular_vel:.2f}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
        cv2.putText(frame, f"Angle Offset: {angle_offset_deg:.2f} deg",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
        cv2.putText(frame, f"User Ctr: ({int(user_center_x)}, {int(user_center_y)})",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        # 2) Draw bounding box if we have one
        if bbox is not None:
            (bx, by, bw, bh) = bbox
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)

        # 3) If Pose is detected, optionally draw a small line to show "torso"
        if pose_detected:
            # Just an example line near user_center_y
            cx = int(user_center_x)
            cy = int(user_center_y)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 2)


# ----------------------------
#             MAIN
# ----------------------------
if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        distance_arg = float(args[1])
    else:
        distance_arg = 0.5

    if not (0 < distance_arg < 2):
        distance_arg = 0.5

    tracker = DistanceAngleTracker(
        camera_index=0,
        target_distance=0.5,         # Desired distance to maintain
        reference_distance=distance_arg,  # Known distance for calibration
        polling_interval=0.1,
        port='/dev/ttyUSB0',
        baudrate=9600,
        serial_enabled=False,
        draw_enabled=True,
        kv=0.8,
        kw=0.005,
    )

    # First, calibrate reference height
    tracker.calibrate_reference()

    # Start the main tracking loop
    tracker.start_tracking()

    print("Ending Script")

                    
 
