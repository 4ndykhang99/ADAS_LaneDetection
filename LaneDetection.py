import cv2
import numpy as np

def region_of_interest(img, vertices):
    """
    Apply the ROI mask to the image.
    """
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def average_slope_intercept(lines):
    """
    Separate lines by slope (left and right) and compute the average slope and intercept.
    Returns two tuples (slope, intercept) for the left and right lane lines.
    """
    left_lines = []
    right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 - x1 == 0:  # avoid division by zero
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            # Assume the left lane has a negative slope, and the right lane has a positive slope
            if slope < 0:
                left_lines.append((slope, intercept, length))
            else:
                right_lines.append((slope, intercept, length))
    
    left_lane = None
    right_lane = None
    if len(left_lines) > 0:
        left_lane = np.average(np.array(left_lines), axis=0, weights=[line[2] for line in left_lines])
    if len(right_lines) > 0:
        right_lane = np.average(np.array(right_lines), axis=0, weights=[line[2] for line in right_lines])
    
    return left_lane, right_lane

def make_line_points(y1, y2, line):
    """
    Based on the (slope, intercept) of the line, calculate the starting and ending points from y1 to y2.
    Returns None if the slope is zero (or nearly zero) to avoid division by zero.
    """
    if line is None:
        return None
    slope, intercept = line[0], line[1]
    # Check if slope is nearly zero to avoid division by zero
    if abs(slope) < 1e-6:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, int(y1)), (x2, int(y2))

def draw_polygon(img, left_line, right_line):
    """
    Draw a polygon connecting the two lane lines, representing the vehicle's path.
    """
    height = img.shape[0]
    y_bottom = height
    y_top = int(height * 0.6)
    left_points = make_line_points(y_bottom, y_top, left_line)
    right_points = make_line_points(y_bottom, y_top, right_line)
    if left_points is None or right_points is None:
        return img
    # Polygon points: bottom left, bottom right, top right, top left
    polygon_points = np.array([left_points[0], right_points[0], right_points[1], left_points[1]])
    cv2.fillPoly(img, [polygon_points], (0, 255, 0))
    return img

def process_frame(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform Canny edge detection
    canny = cv2.Canny(gaussian, 50, 150)
    
    # Define the dimensions of the image and the ROI (assume the lower part of the image)
    height, width = canny.shape
    roi_vertices = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.45), int(height * 0.6))
    ]], dtype=np.int32)
    roi = region_of_interest(canny, roi_vertices)
    
    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=150)
    
    left_lane, right_lane = None, None
    if lines is not None:
        left_lane, right_lane = average_slope_intercept(lines)
    
    # Calculate line points from y = height to y = 0.6 * height
    left_line_points = make_line_points(height, int(height * 0.6), left_lane)
    right_line_points = make_line_points(height, int(height * 0.6), right_lane)
    
    # Create a copy of the original frame to draw the results
    final_result = frame.copy()
    
    # Draw the detected lane lines (blue color)
    if left_line_points is not None:
        cv2.line(final_result, left_line_points[0], left_line_points[1], (255, 0, 0), 10)
    if right_line_points is not None:
        cv2.line(final_result, right_line_points[0], right_line_points[1], (255, 0, 0), 10)
    
    # Draw the polygon representing the vehicle's path (green color)
    final_result = draw_polygon(final_result, left_lane, right_lane)
    
    return gaussian, canny, roi, final_result

def main():
    # Use video from a file or camera (change "test_video.mp4" to your video path,
    # or use cv2.VideoCapture(0) to use the webcam)
    cap = cv2.VideoCapture("Driving_Vid2.mp4")
    
    if not cap.isOpened():
        print("Unable to open video. Please check the video path or camera connection.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or unable to read frame.")
            break
        
        # Process each frame
        gaussian, canny, roi, final_result = process_frame(frame)
        
        # Resize window
        # gaussian = cv2.resize(gaussian, (0, 0), None, .25, .25)
        # canny = cv2.resize(canny, (0, 0), None, .25, .25)
        # roi = cv2.resize(roi, (0, 0), None, .25, .25)
        # final_result = cv2.resize(final_result, (0, 0), None, .25, .25)

        # Display the results in 4 separate windows
        # cv2.imshow("Gaussian Blur", gaussian_rz)
        # cv2.imshow("Canny Edges", canny_rz)
        # cv2.imshow("Region of Interest (ROI)", roi_rz)
        # cv2.imshow("Final Result", final_result_rz)

        # Resize images
        gaussian = cv2.resize(gaussian, (0, 0), fx=0.25, fy=0.25)
        canny = cv2.resize(canny, (0, 0), fx=0.25, fy=0.25)
        roi = cv2.resize(roi, (0, 0), fx=0.25, fy=0.25)
        final_result = cv2.resize(final_result, (0, 0), fx=0.25, fy=0.25)

        # Check and revert grayscale to BGR if needed
        if len(gaussian.shape) == 2:
            gaussian = cv2.cvtColor(gaussian, cv2.COLOR_GRAY2BGR)
        if len(canny.shape) == 2:
            canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        if len(roi.shape) == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        if len(final_result.shape) == 2:
            final_result = cv2.cvtColor(final_result, cv2.COLOR_GRAY2BGR)

        # combine window
        top_row = np.hstack((gaussian, canny))
        bottom_row = np.hstack((roi, final_result))
        combined = np.vstack((top_row, bottom_row))

        # Show the result
        cv2.imshow("Combined Results", combined)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
