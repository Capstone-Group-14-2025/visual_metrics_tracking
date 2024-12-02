import cv2
import time
import numpy as np

def visual_tracking(self, print_type="movement"):
        while True:
            current_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                print("Could not read frame.")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if self.klt_tracker_initialized and self.feature_points is not None:
                # Convert frame to grayscale for KLT tracking
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Use KLT tracker to find new positions of feature points
                new_feature_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray_frame, gray_frame, self.feature_points, None, **self.lk_params
                )

                # Filter valid points (where status is 1)
                valid_points = new_feature_points[status == 1]
                if len(valid_points) > 0:
                    avg_x = np.mean(valid_points[:, 0])
                    avg_y = np.mean(valid_points[:, 1])
                    horizontal_offset = avg_x - frame.shape[1] / 2

                    # Estimate distance using tracked points
                    distance = self.estimate_distance(abs(valid_points[0][1] - avg_y))
                    self.control_movement(distance, horizontal_offset, frame.shape[1], print_type=print_type)

                    # Draw the tracked points
                    for point in valid_points:
                        cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

                # Update feature points and previous frame
                self.feature_points = new_feature_points
                self.prev_gray_frame = gray_frame

            else:
                # First frame grayscale conversion for KLT tracker initialization
                self.prev_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow("User Tracking", frame)

        self.cap.release()
        cv2.destroyAllWindows()
