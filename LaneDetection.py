import cv2
import numpy as np

def region_of_interest(img, vertices):
    """
    Áp dụng mặt nạ cho ảnh, chỉ giữ lại vùng xác định bởi vertices.
    """
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def make_line_points(frame, line):
    """
    Nội suy các điểm của đường thẳng từ hệ số góc (slope) và hệ số chặn (intercept).
    Trả về điểm đầu và điểm cuối của đường thẳng để vẽ.
    """
    slope, intercept = line
    y1 = frame.shape[0]       # đáy ảnh
    y2 = int(y1 * 0.6)          # một phần phía trên đáy ảnh (điểm cuối của đường)
    # Tính điểm x từ phương trình đường thẳng: x = (y - intercept) / slope
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, y1, x2, y2]

def compute_average_lines(frame, lines):
    """
    Phân tách các đường thành nhóm bên trái và bên phải dựa trên độ dốc.
    Tính trung bình các đường trong mỗi nhóm để có được đường ước tính cho làn trái và làn phải.
    """
    left_lines = []
    right_lines = []
    if lines is None:
        return None, None
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 - x1 == 0:
                continue  # tránh chia cho 0
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            # Đường bên trái có độ dốc âm, bên phải có độ dốc dương (có thể điều chỉnh ngưỡng tùy thuộc vào video)\n
            if slope < -0.5:
                left_lines.append((slope, intercept))
            elif slope > 0.5:
                right_lines.append((slope, intercept))
    left_line = None
    right_line = None
    if left_lines:
        left_avg = np.mean(left_lines, axis=0)
        left_line = make_line_points(frame, left_avg)
    if right_lines:
        right_avg = np.mean(right_lines, axis=0)
        right_line = make_line_points(frame, right_avg)
    return left_line, right_line

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video:", video_path)
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Tiền xử lý: chuyển sang grayscale, làm mờ Gaussian và phát hiện cạnh bằng Canny
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Xác định vùng quan tâm (ROI) - giả sử là vùng hình tứ giác chứa làn xe
        height, width = frame.shape[:2]
        roi_vertices = np.array([[
            (0, height),
            (width, height),
            (int(width * 0.55), int(height * 0.6)),
            (int(width * 0.45), int(height * 0.6))
        ]], np.int32)
        masked_edges = region_of_interest(edges, roi_vertices)

        # Dùng Hough Transform để phát hiện các đường thẳng
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=40, maxLineGap=100)

        # Tính toán trung bình và nội suy cho các đường làn trái và phải
        left_line, right_line = compute_average_lines(frame, lines)

        # Nếu cả hai đường làn đều được xác định, tạo đa giác bao quanh làn xe
        if left_line is not None and right_line is not None:
            # Các điểm của đa giác: điểm dưới bên trái, trên bên trái, trên bên phải, dưới bên phải
            polygon_points = np.array([
                [left_line[0], left_line[1]],   # đáy trái
                [left_line[2], left_line[3]],   # trên trái
                [right_line[2], right_line[3]], # trên phải
                [right_line[0], right_line[1]]  # đáy phải
            ], np.int32)
            polygon_points = polygon_points.reshape((-1, 1, 2))
            
            # Tạo overlay để vẽ đa giác với độ trong suốt (alpha)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon_points], (0, 255, 0))
            alpha = 0.3  # độ trong suốt của overlay
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Vẽ thêm các đường viền của làn xe (tùy chọn) để dễ nhận biết\n
            cv2.line(frame, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (255, 0, 0), 10)
            cv2.line(frame, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (255, 0, 0), 10)

        cv2.imshow("Lane Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "Driving_Vid2.mp4"  # Đổi tên hoặc đường dẫn video của bạn ở đây
    process_video(video_path)
