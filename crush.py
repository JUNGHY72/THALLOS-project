import cv2
import numpy as np

# ⚙️ 다각형 마스크 생성
def create_mask_from_polygon(img_shape, polygon):
    mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask

# ⚙️ HSV 기반 흰색 필터
def filter_white(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    return cv2.inRange(hsv, lower_white, upper_white)

# ⚙️ (y, x) 기반 2차 곡선 피팅
def fit_curve(points, y_top, y_bottom):
    if len(points) >= 5:
        points = np.array(points)
        fit = np.polyfit(points[:, 1], points[:, 0], 2)  # x = f(y)
        y_vals = np.linspace(y_top, y_bottom, 50)
        x_vals = np.polyval(fit, y_vals)
        return np.vstack((x_vals, y_vals)).T.astype(np.int32)
    return None

# ⚙️ 곡선 좌우 대칭
def mirror_curve_across_center(curve, center_x):
    if curve is None:
        return None
    mirrored = curve.copy()
    mirrored[:, 0] = 2 * center_x - curve[:, 0]
    return mirrored

# ⚙️ 곡선 점프 여부
def is_jump(prev, new, threshold=50):
    if prev is None or new is None or prev.shape != new.shape:
        return False
    dx = np.abs(prev[:, 0] - new[:, 0])
    return np.any(dx > threshold)

# 🎥 비디오 열기
cap = cv2.VideoCapture("/home/hkit/다운로드/test_movie_009.mp4")
ret, frame = cap.read()
if not ret:
    print("영상 로드 실패")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 33

frame_height, frame_width = frame.shape[:2]
center_x = frame_width // 2

# 📐 ROI 설정
y_top_fixed = int(frame_height * 0.6)
y_bottom_fixed = frame_height

zone_caution = np.array([
    [int(frame_width * 0.43), y_top_fixed],
    [int(frame_width * 0.57), y_top_fixed],
    [int(frame_width * 0.85), y_bottom_fixed],
    [int(frame_width * 0.15), y_bottom_fixed]
], np.int32)

mask = create_mask_from_polygon(frame.shape, zone_caution)

# 🎯 상태 초기화
prev_left_curve = None
prev_right_curve = None
alpha = 0.2
lost_frame_count_left = 0
lost_frame_count_right = 0
max_lost_frames = 5

# ▶️ 메인 루프
while True:
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask)
    white_mask = filter_white(roi_frame)
    white_only = cv2.bitwise_and(roi_frame, roi_frame, mask=white_mask)
    white_gray = cv2.cvtColor(white_only, cv2.COLOR_BGR2GRAY)
    white_enhanced = cv2.convertScaleAbs(white_gray, alpha=2.0, beta=0)

    coords = cv2.findNonZero(white_enhanced)
    left_pts, right_pts = [], []

    if coords is not None:
        for pt in coords:
            x, y = pt[0]
            if y_top_fixed <= y <= y_bottom_fixed:
                (left_pts if x < center_x else right_pts).append([x, y])

    new_left_curve = fit_curve(left_pts, y_top_fixed, y_bottom_fixed)
    new_right_curve = fit_curve(right_pts, y_top_fixed, y_bottom_fixed)

    if new_left_curve is None and new_right_curve is not None:
        new_left_curve = mirror_curve_across_center(new_right_curve, center_x)
    elif new_right_curve is None and new_left_curve is not None:
        new_right_curve = mirror_curve_across_center(new_left_curve, center_x)

    if new_left_curve is not None and not is_jump(prev_left_curve, new_left_curve):
        left_curve = ((1 - alpha) * prev_left_curve + alpha * new_left_curve).astype(np.int32) \
            if prev_left_curve is not None and len(prev_left_curve) == len(new_left_curve) \
            else new_left_curve
        prev_left_curve = left_curve
        lost_frame_count_left = 0
    else:
        lost_frame_count_left += 1
        left_curve = prev_left_curve if lost_frame_count_left <= max_lost_frames else None

    if new_right_curve is not None and not is_jump(prev_right_curve, new_right_curve):
        right_curve = ((1 - alpha) * prev_right_curve + alpha * new_right_curve).astype(np.int32) \
            if prev_right_curve is not None and len(prev_right_curve) == len(new_right_curve) \
            else new_right_curve
        prev_right_curve = right_curve
        lost_frame_count_right = 0
    else:
        lost_frame_count_right += 1
        right_curve = prev_right_curve if lost_frame_count_right <= max_lost_frames else None

    if left_curve is None and right_curve is not None:
        left_curve = mirror_curve_across_center(right_curve, center_x)
    if right_curve is None and left_curve is not None:
        right_curve = mirror_curve_across_center(left_curve, center_x)

    output = frame.copy()

    # 🟩 사다리꼴 그리기 조건
    if left_curve is not None and right_curve is not None:
        left_sorted = left_curve[np.argsort(left_curve[:, 1])]
        right_sorted = right_curve[np.argsort(right_curve[:, 1])]

        min_len = min(len(left_sorted), len(right_sorted))
        left_sorted = left_sorted[:min_len]
        right_sorted = right_sorted[:min_len]

        top_width = np.linalg.norm(left_sorted[0] - right_sorted[0])
        bottom_width = np.linalg.norm(left_sorted[-1] - right_sorted[-1])
        if bottom_width < top_width or left_sorted[0][0] > right_sorted[0][0]:
            ret, frame = cap.read()
            if not ret or cv2.waitKey(delay) & 0xFF == ord('q'):
                break
            continue

        # 🔽 하단 고정 좌표 덮기
        left_sorted[-1][0] = int(frame_width * 0.3)
        right_sorted[-1][0] = int(frame_width * 0.7)

        polygon_pts = np.array([
            *left_sorted,
            *right_sorted[::-1]
        ]).reshape((-1, 1, 2))

        # 반투명 채우기
        overlay = output.copy()
        cv2.fillPoly(overlay, [polygon_pts], color=(0, 255, 0))
        cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)

        # 외곽선
        cv2.polylines(output, [polygon_pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 🎨 중간 결과 시각화
    debug_masked = cv2.bitwise_and(frame, frame, mask=mask)
    debug_white = cv2.bitwise_and(debug_masked, debug_masked, mask=white_mask)

    cv2.imshow("ROI Only", debug_masked)
    cv2.imshow("White Filtered", white_mask)
    cv2.imshow("White Only", debug_white)
    cv2.imshow("White Gray", white_gray)
    cv2.imshow("White Enhanced", white_enhanced)

    lane_points = frame.copy()
    for x, y in left_pts:
        cv2.circle(lane_points, (x, y), 2, (255, 0, 0), -1)
    for x, y in right_pts:
        cv2.circle(lane_points, (x, y), 2, (0, 0, 255), -1)
    cv2.imshow("Lane Points", lane_points)

    # 🖼️ 최종 결과 출력
    cv2.imshow("Lane Detection - Filled Bottom", output)

    # ⏭️ 다음 프레임
    ret, frame = cap.read()
    if not ret or cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
