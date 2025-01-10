# Import libraries
import cv2
import numpy as np

# Initialize SIFT detector
sift = cv2.SIFT_create()

FLANN_INDEX_KDTREE = 1
#Specifies the number of k-d trees to be used for building the index
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) 
#Sets the number of times the tree will be traversed during the search
search_params = dict(checks=50)
#Create a FLANN-based matcher object using the specified parameters
flann = cv2.FlannBasedMatcher(index_params, search_params)


# frame is the input image where object tracking begins. Typically a single frame from a video or an image.
# bbox = A bounding box specifying the region of interest (ROI) for tracking in the form (x, y, w, h)
def initialize_tracking(frame, bbox):
    # x, y = top-left corner coordinates of the bounding box
    # w, h  = Width and height of the bounding box
    x, y, w, h = bbox 
    # extract the rectangular region in the frame defined by the bounding box bbox
    object_roi = frame[y:y+h, x:x+w]
    
    """
    Detects keypoints in the entire frame.
    Computes descriptors for these keypoints.
    keypoints: List of keypoints (points of interest) in the image.
    descriptors: Corresponding feature descriptors (local image patches around the keypoints).
    """
    keypoints, descriptors = sift.detectAndCompute(frame, None)
    object_keypoints = []
    object_descriptors = []
    background_keypoints = []
    background_descriptors = []
    
    """
    kp.pt: The (x, y) coordinates of a keypoint.
    Condition:
        If the keypoint lies within the bounding box, it is classified as an object keypoint.
        Otherwise, it is classified as a background keypoint.
        desc: The descriptor for the current keypoint, added to the appropriate list based on location.

    """

    for kp, desc in zip(keypoints, descriptors):
        if x <= kp.pt[0] <= x+w and y <= kp.pt[1] <= y+h:
            object_keypoints.append(kp)
            object_descriptors.append(desc)
        else:
            background_keypoints.append(kp)
            background_descriptors.append(desc)

    object_descriptors = np.array(object_descriptors)
    background_descriptors = np.array(background_descriptors)

    return object_keypoints, object_descriptors, background_keypoints, background_descriptors, (x, y, w, h)



def track_object(frame, object_descriptors, background_descriptors, prev_bbox):
    keypoints, descriptors = sift.detectAndCompute(frame, None)
    
    # Perform FLANN-based matching
    
    #tuple with matches (x and y pairs)
    object_matches = flann.knnMatch(descriptors, object_descriptors, k=2)
    background_matches = flann.knnMatch(descriptors, background_descriptors, k=2)

    object_confidences = []
    object_points = []
    """
    For each keypoint:
    Compute the distance ratio:
    do: Distance to the best match (object)
    db: Distance to the second-best match (background)
    Confidence is positive (1) if the ratio test passes; otherwise, negative (-1)
    
    """
    for i, kp in enumerate(keypoints):
        if i < len(object_matches) and len(object_matches[i]) == 2:
            do = object_matches[i][0].distance
            db = object_matches[i][1].distance if len(background_matches[i]) == 2 else float('inf')
            confidence = 1 if do / db < 0.5 else -1
            object_confidences.append((kp, confidence))
            if confidence == 1:
                object_points.append(kp.pt)

    if len(object_points) > 0:
        # Use RANSAC to estimate a more reliable bounding box
        points = np.array(object_points)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        new_bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
    else:
        new_bbox = prev_bbox

    return object_confidences, new_bbox

def main():
    # Open webcam feed
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam. Change index if necessary.

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Unable to read the first frame from the webcam.")
        return

    # Select ROI
    print("Select a ROI and press SPACE or ENTER. Cancel the selection by pressing 'C'.")
    bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    if bbox == (0, 0, 0, 0):  # Check for empty ROI
        print("Error: No valid ROI selected.")
        return
    cv2.destroyWindow("Select ROI")

    # Initialize object features using SIFT
    object_keypoints, object_descriptors, background_keypoints, background_descriptors, bbox = initialize_tracking(frame, bbox)

    # Initialize the tracker (CSRT) 
    # TO DO: understand mathematical details of this.
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    # Main loop for tracking
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the webcam.")
            break

        # Update tracker (CSRT) 
      
        success, bbox = tracker.update(frame)

        # Use SIFT to validate and adjust tracking if necessary
        if not success:
            print("CSRT tracking failed, trying SIFT matching...")
            object_confidences, new_bbox = track_object(frame, object_descriptors, background_descriptors, bbox)
            if object_confidences:
                # If confident matches are found, update the bounding box
                x, y, w, h = new_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                bbox = new_bbox  # Update the bounding box for future use
                print("SIFT matching...")
            else:
                cv2.putText(frame, "Tracking failed!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        else:
            # Draw the bounding box for successful CSRT tracking
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the result
        cv2.imshow("Tracking", frame)

        # Exit on pressing 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
