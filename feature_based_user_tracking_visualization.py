import cv2
import numpy as np

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Initialize FLANN-based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def draw_matches_frame(frame1, kp1, frame2, kp2, matches):
    """Draw matches between two frames"""
    # Create a combined image
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    
    # Create output image
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = frame1
    vis[:h2, w1:w1+w2] = frame2

    # Draw matches
    for match in matches:
        # Get the matching keypoints for each of the images
        if match.queryIdx >= len(kp1) or match.trainIdx >= len(kp2):
            continue
            
        # Get the coordinates
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        # x - columns, y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw the matches
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2 + w1), int(y2))
        
        cv2.circle(vis, pt1, 3, (0, 255, 0), -1)
        cv2.circle(vis, pt2, 3, (0, 255, 0), -1)
        cv2.line(vis, pt1, pt2, (0, 255, 0), 1)

    return vis

def initialize_tracking(frame, bbox):
    x, y, w, h = bbox
    # Store initial frame for match visualization
    global initial_frame, initial_keypoints, initial_descriptors
    initial_frame = frame.copy()
    
    # Detect keypoints and compute descriptors
    initial_keypoints, initial_descriptors = sift.detectAndCompute(frame, None)
    
    # Draw initial keypoints
    keypoints_vis = cv2.drawKeypoints(frame.copy(), initial_keypoints, None, 
                                    color=(0, 255, 0), 
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Initial Keypoints", keypoints_vis)
    
    # Filter object keypoints
    object_keypoints = []
    object_descriptors = []
    background_keypoints = []
    background_descriptors = []
    
    for kp, desc in zip(initial_keypoints, initial_descriptors):
        if x <= kp.pt[0] <= x+w and y <= kp.pt[1] <= y+h:
            object_keypoints.append(kp)
            object_descriptors.append(desc)
        else:
            background_keypoints.append(kp)
            background_descriptors.append(desc)
    
    if len(object_descriptors) > 0:
        object_descriptors = np.array(object_descriptors)
    if len(background_descriptors) > 0:
        background_descriptors = np.array(background_descriptors)
    
    return object_keypoints, object_descriptors, background_keypoints, background_descriptors, (x, y, w, h)

def track_object(frame, object_descriptors, background_descriptors, prev_bbox):
    current_keypoints, current_descriptors = sift.detectAndCompute(frame, None)
    
    if len(current_descriptors) == 0 or len(initial_descriptors) == 0:
        return [], prev_bbox

    try:
        # Match features with initial frame
        matches = flann.knnMatch(current_descriptors, initial_descriptors, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # Draw matches visualization
        if len(good_matches) > 0:
            matches_vis = draw_matches_frame(initial_frame, initial_keypoints, 
                                           frame, current_keypoints, 
                                           good_matches)
            cv2.imshow("Feature Matches", matches_vis)
    except Exception as e:
        print(f"Matching error: {e}")
        good_matches = []

    # Calculate confidence for tracking
    object_confidences = []
    for kp in current_keypoints:
        object_confidences.append((kp, 1))  # Simplified confidence

    x, y, w, h = prev_bbox
    return object_confidences, (x, y, w, h)

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Unable to read the first frame from the webcam.")
        return

    print("Select a ROI and press SPACE or ENTER. Cancel the selection by pressing 'C'.")
    bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    if bbox == (0, 0, 0, 0):
        print("Error: No valid ROI selected.")
        return
    cv2.destroyWindow("Select ROI")

    # Initialize tracking
    object_kp, object_desc, bg_kp, bg_desc, bbox = initialize_tracking(frame, bbox)
    
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the webcam.")
            break

        # Update tracker
        success, bbox = tracker.update(frame)
        
        # Update feature tracking and get visualization
        confidences, new_bbox = track_object(frame, object_desc, bg_desc, bbox)

        # Draw tracking result
        result_frame = frame.copy()
        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.putText(result_frame, "Tracking failed!", (50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow("Tracking Result", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
