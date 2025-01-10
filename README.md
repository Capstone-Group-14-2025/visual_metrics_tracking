Feature based user tracking

The following implemention uses Scale-Invariant Feature Transform detector for accurate user tracking

SIFT is a feature detection algorithm used in computer vision to identify and describe distinctive points in images (keypoints).

It is scale and rotation invariant, meaning it can detect the same features even if the image is resized or rotated.
Steps in SIFT:

Scale-Space Extrema Detection: Identify potential keypoints using a Difference of Gaussian (DoG) method across multiple scales.

Keypoint Localization: Refine these points to filter out noise.

Orientation Assignment: Assign orientations to keypoints to make them rotation-invariant.

Keypoint Descriptor: Generate a unique descriptor vector for each keypoint.
