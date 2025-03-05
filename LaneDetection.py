import cv2
import numpy as np
import os

def region_of_interest(img, vertices):
    """
    Apply a mask to the region of interest (ROI) defined by the given vertices.
    """
    mask = np.zeros_like(img)
    # For color images, set the mask color accordingly.
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[0, 255, 0], thickness=10):
    """
    Draw the detected lines on the image.
    """
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def process_frame(frame):
    """
    Process each frame by applying the following steps:
    1. Convert to grayscale.
    2. Apply Gaussian blur.
    3. Perform Canny edge detection.
    4. Apply region of interest mask.
    5. Detect lane lines using Hough Transform.
    6. Overlay the detected lane markings on the original frame.
    
    Returns:
        gray_bgr: Grayscale image converted back to BGR.
        blur_bgr: Blurred image converted back to BGR.
        edges_bgr: Canny edge detection result in BGR.
        roi_bgr: Image after applying ROI mask in BGR.
        hough_img: Image with Hough lines drawn.
        final_output: Final overlay of detected lanes on the original frame.
    """
    # 1. Convert to grayscale and convert back to BGR for saving
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # 2. Apply Gaussian blur and convert to BGR
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur_bgr = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    
    # 3. Canny edge detection and convert to BGR
    edges = cv2.Canny(blur, 50, 150)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 4. Define region of interest (ROI) - usually the lower part of the frame
    height, width = frame.shape[:2]
    vertices = np.array([[
        (50, height),
        (width / 2 - 40, height / 2 + 50),
        (width / 2 + 40, height / 2 + 50),
        (width - 50, height)
    ]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    
    # 5. Detect lane lines using Hough Transform
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )
    # Create an image to draw the Hough lines
    hough_img = np.zeros_like(frame)
    draw_lines(hough_img, lines)
    
    # 6. Overlay the detected lines on the original frame
    final_output = cv2.addWeighted(frame, 0.8, hough_img, 1, 1)
    
    return gray_bgr, blur_bgr, edges_bgr, roi_bgr, hough_img, final_output

def main():
    video_path = "Driving_Vid.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Unable to open video")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create the Output folder if it doesn't exist
    os.makedirs("Output", exist_ok=True)
    
    # Define output paths for each stage
    output_paths = {
        "grayscale": os.path.join("Output", "grayscale.mp4"),
        "gaussian": os.path.join("Output", "gaussian.mp4"),
        "canny": os.path.join("Output", "canny.mp4"),
        "roi": os.path.join("Output", "roi.mp4"),
        "hough": os.path.join("Output", "hough_lines.mp4"),
        "final": os.path.join("Output", "final_output.mp4")
    }
    
    # Create VideoWriter objects for each stage
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_grayscale = cv2.VideoWriter(output_paths["grayscale"], fourcc, fps, (width, height))
    out_gaussian = cv2.VideoWriter(output_paths["gaussian"], fourcc, fps, (width, height))
    out_canny = cv2.VideoWriter(output_paths["canny"], fourcc, fps, (width, height))
    out_roi = cv2.VideoWriter(output_paths["roi"], fourcc, fps, (width, height))
    out_hough = cv2.VideoWriter(output_paths["hough"], fourcc, fps, (width, height))
    out_final = cv2.VideoWriter(output_paths["final"], fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame to get all stages
        gray_frame, blur_frame, edges_frame, roi_frame, hough_frame, final_frame = process_frame(frame)
        
        # Write each stage to its corresponding video file
        out_grayscale.write(gray_frame)
        out_gaussian.write(blur_frame)
        out_canny.write(edges_frame)
        out_roi.write(roi_frame)
        out_hough.write(hough_frame)
        out_final.write(final_frame)
        
        # Optionally, display the final output
        cv2.imshow('Final Lane Detection', final_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out_grayscale.release()
    out_gaussian.release()
    out_canny.release()
    out_roi.release()
    out_hough.release()
    out_final.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
