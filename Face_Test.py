import cv2
import dlib
from math import hypot
import time
import subprocess
from monitorcontrol import get_monitors
monitors = get_monitors()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_SIMPLEX

r_eye_points = [42, 43, 44, 45, 46, 47]
l_eye_points = [36, 37, 38, 39, 40, 41]

count_mouth_open = 0
is_blinking = False
blink_start_time = 0
blink_duration = 0

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(image, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(image, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length / ver_line_length
    return ratio

def setBrightSound(brightness, volume):
    # 모니터 밝기 조절
    if monitors:
        with monitors[0] as monitor:
            # 현재 밝기 확인
            current_brightness = monitor.get_luminance()
            print(f"현재 밝기: {current_brightness}%")

            # 밝기 설정 (0-100 사이의 값)
            new_brightness = brightness  # 원하는 밝기 값
            monitor.set_luminance(new_brightness)
            print(f"밝기를 {new_brightness}%로 설정했습니다.")
    else:
        print("모니터를 찾을 수 없습니다.")

    # 소리 조절
    # 0-100 볼륨 값에서 NirCmd의 범위로 변환
    volume_value = int(volume * 0.01 * 65536)  # 볼륨 값 계산 (정수 변환)
    
    # NirCmd 실행하여 볼륨 조절 (-volume_value로 소리 줄임)
    subprocess.run(["cmd", "/c", f"nircmd.exe changesysvolume {-volume_value}"])




capture = cv2.VideoCapture(0) 
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


blink_start_time = time.time()  # 눈을 감기 시작한 시간 기록
blink_end_time = time.time()

while True:
    _, image = capture.read()

    # convert frame to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_ratio = get_blinking_ratio(l_eye_points, landmarks)
        right_eye_ratio = get_blinking_ratio(r_eye_points, landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio >= 6.0:  # 눈 감았을 때
            if not is_blinking:  # 처음 눈을 감았을 때만 시간 기록
                is_blinking = True
                blink_end_time = time.time()
                print(f"Blinking Time: {blink_duration:.2f} sec")
                cv2.putText(image, "blinking", (50, 50), font, 2, (255, 0, 0))
                print("blinking")
                setBrightSound(20, 40)
  
                        
        else:  # 눈을 뜨고 있을 때
            if is_blinking:  # 눈을 감고 있다가 뜰 때만 시간 계산
                is_blinking = False
                blink_duration = blink_end_time - blink_start_time  # 깜빡임 지속시간 계산
                print(f"Blinking Time: {blink_duration:.2f} sec")
                cv2.putText(image, f"Blinking duration: {blink_duration:.2f}s", (50, 100), font, 1, (0, 255, 0))
                blink_start_time = time.time()  # 눈을 감기 시작한 시간 기록

    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()