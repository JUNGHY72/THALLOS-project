import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------- 설정 ---------------------------

# 모델 로드
model = YOLO("/home/hkit/바탕화면/yolov8_custom14/weights/best.pt")

# 영상 로드
cap = cv2.VideoCapture("/home/hkit/바탕화면/LotteWorldTower_10to19min.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 클래스별 실제 높이(mm)
CLASS_HEIGHTS_MM = {
    "vehicle": 1500,
    "bigvehicle": 3000,
    "bike": 1200,
    "human": 1700,
    "animal": 800,
    "obstacle": 1000
}

# ---------------------- 함수 정의 -----------------------

# 사다리꼴 영역 계산 함수
def get_trapezoid_zone(top_ratio, bottom_ratio, height_ratio):
    top_width = int(frame_width * top_ratio)
    bottom_width = int(frame_width * bottom_ratio)
    height = int(frame_height * height_ratio)

    top_left = ((frame_width - top_width) // 2, frame_height - height)
    top_right = ((frame_width + top_width) // 2, frame_height - height)
    bottom_left = ((frame_width - bottom_width) // 2, frame_height)
    bottom_right = ((frame_width + bottom_width) // 2, frame_height)

    return np.array([top_left, top_right, bottom_right, bottom_left], np.int32)

# 사다리꼴 영역 정의
zone_warning = get_trapezoid_zone(0.35, 0.6, 0.2)
zone_caution = get_trapezoid_zone(0.2, 0.9, 0.35)

# 박스와 영역 겹침 여부
def is_box_overlapping_zone(box_coords, zone_polygon):
    x1, y1, x2, y2 = box_coords
    box_polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    intersection = cv2.intersectConvexConvex(box_polygon.astype(np.float32), zone_polygon.astype(np.float32))
    return intersection[0] > 0  # 면적이 0보다 크면 겹침

# 카메라 파라미터 기반 거리 추정
def estimate_distance_by_focal_length(box_height_px, class_name, focal_length_mm=35.0, sensor_height_mm=23.9):
    if box_height_px == 0:
        return float('inf')
    real_height_mm = CLASS_HEIGHTS_MM.get(class_name, 1700)
    distance_mm = (focal_length_mm * real_height_mm * frame_height) / (box_height_px * sensor_height_mm)
    return round(distance_mm / 1000, 1)

# y 좌표 기반 거리 추정
def estimate_distance_by_y_position(y2_pixel):
    norm_y = (frame_height - y2_pixel) / frame_height
    estimated_distance_m = norm_y * 10
    return round(estimated_distance_m, 1)

# ---------------------- 메인 루프 -----------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 객체 탐지 수행
    results = model(frame, verbose=False)
    result = results[0]

    # 사다리꼴 영역 표시
    overlay = frame.copy()
    cv2.polylines(overlay, [zone_caution], True, (0, 255, 255), 2)
    cv2.polylines(overlay, [zone_warning], True, (0, 0, 255), 2)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # 객체 탐지 결과 처리
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_h = y2 - y1
            box_coords = (x1, y1, x2, y2)

            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            # 거리 추정 (두 방식 평균)
            distance_box = estimate_distance_by_focal_length(box_h, class_name)
            distance_y = estimate_distance_by_y_position(y2)
            avg_distance = (distance_box + distance_y) / 2
            distance_m = round(avg_distance * 0.5, 1)  # 시각화용 조정

            # 경고/주의 판단
            if is_box_overlapping_zone(box_coords, zone_warning):
                color = (0, 0, 255)
                label = f"WARNING! {class_name} {distance_m}m"
            elif is_box_overlapping_zone(box_coords, zone_caution):
                color = (0, 255, 255)
                label = f"CAUTION: {class_name} {distance_m}m"
            else:
                color = (0, 255, 0)
                label = f"{class_name} {distance_m}m"

            # 바운딩 박스 및 라벨 출력
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 프레임 출력
    cv2.imshow("Distance Warning", frame)

    # 종료 키
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------- 종료 처리 -----------------------
cap.release()
cv2.destroyAllWindows()
