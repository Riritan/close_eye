import cv2
import mediapipe as mp
import time
from close_eye_control_test import setBrightSound
from initMonitor import InitMonitor

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

capture = cv2.VideoCapture(0)

# 색상 정의
red_color = (0, 0, 255)
green_color = (0, 255, 0)

# 조도와 소리 기본 값 (100%)
brightness = 100
sound = 100

set_level = [100, 80, 30, 10, 0]
current_set_level = 0

is_off = False
is_set_monitor = False

InitMonitor()

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.5) as face_mesh:

    blink_start_time = 0
    now_blink = False

    while capture.isOpened():
        success, image = capture.read()
        if not success:
            print("웹캠을 찾을 수 없습니다")
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        face_results = face_detection.process(image)
        mesh_results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                # 왼쪽 및 오른쪽 눈 랜드마크 좌표
                left_eyes = [159, 145]
                right_eyes = [386, 374]

                # 왼쪽 눈 및 오른쪽 눈 거리 계산
                left_distance = abs(face_landmarks.landmark[left_eyes[0]].y - face_landmarks.landmark[left_eyes[1]].y) * 100
                right_distance = abs(face_landmarks.landmark[right_eyes[0]].y - face_landmarks.landmark[right_eyes[1]].y) * 100

                # 눈 사각형 위치 설정
                left_eye_top_left = (int(face_landmarks.landmark[left_eyes[0]].x * image.shape[1] - 40),
                                     int(face_landmarks.landmark[left_eyes[0]].y * image.shape[0]) - 20)
                left_eye_bottom_right = (int(face_landmarks.landmark[left_eyes[1]].x * image.shape[1] + 40),
                                         int(face_landmarks.landmark[left_eyes[1]].y * image.shape[0]) + 20)
                
                right_eye_top_left = (int(face_landmarks.landmark[right_eyes[0]].x * image.shape[1] - 40),
                                      int(face_landmarks.landmark[right_eyes[0]].y * image.shape[0]) - 20)
                right_eye_bottom_right = (int(face_landmarks.landmark[right_eyes[1]].x * image.shape[1] + 40),
                                          int(face_landmarks.landmark[right_eyes[1]].y * image.shape[0]) + 20)

                # 눈이 감긴 상태 (임계값 이하)
                if left_distance < 5.5 and right_distance < 5.5:
                    # 눈 감김 상태가 처음 감지되었을 때 타이머 시작
                    if not now_blink:
                        blink_start_time = time.time()
                        now_blink = True
                    
                    blink_duration = time.time() - blink_start_time

                    # 각 CASE에 따라 조도와 소리 감소
                    if not is_off:
                        if blink_duration >= 15 and current_set_level == 3:
                            current_set_level = 4
                            is_set_monitor = True
                            # brightness, sound = 0, 0  # CASE4: 조도와 소리 100% 감소
                            # setBrightSound(brightness, sound)
                            is_off = True
                        elif blink_duration >= 10 and current_set_level == 2:
                            current_set_level = 3
                            is_set_monitor = True
                            # brightness, sound = 10, 10  # CASE3: 조도와 소리 90% 감소
                            # setBrightSound(brightness, sound)
                        elif blink_duration >= 5 and current_set_level == 1:
                            current_set_level = 2
                            is_set_monitor = True
                            # brightness, sound = 30, 30  # CASE2: 조도와 소리 70% 감소
                            # setBrightSound(brightness, sound)
                        elif blink_duration >= 3 and current_set_level == 0:
                            current_set_level = 1
                            is_set_monitor = True
                            # brightness, sound = 80, 80  # CASE1: 조도와 소리 20% 감소
                            # setBrightSound(brightness, sound)
                        
                        if is_set_monitor:
                            brightness, sound = set_level[current_set_level], set_level[current_set_level]
                            setBrightSound(brightness, sound)
                            is_set_monitor = False

                    # 사각형과 "0" 표시 (눈 감김 상태)
                    cv2.rectangle(image, left_eye_top_left, left_eye_bottom_right, red_color, 1)
                    cv2.putText(image, "0", (left_eye_top_left[0], left_eye_bottom_right[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1)
                    cv2.rectangle(image, right_eye_top_left, right_eye_bottom_right, red_color, 1)
                    cv2.putText(image, "0", (right_eye_top_left[0], right_eye_bottom_right[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1)

                else:
                    # 눈을 뜬 상태로 전환되었을 때 초기화 및 밝기와 소리 원상 복구
                    if now_blink:
                        current_set_level = 0
                        InitMonitor()
                        # brightness, sound = 100, 100
                        # setBrightSound(brightness, sound)
                        is_off = False

                    # 타이머 초기화
                    blink_start_time = time.time()
                    now_blink = False

                    # 사각형과 "1" 표시 (눈 뜸 상태)
                    cv2.rectangle(image, left_eye_top_left, left_eye_bottom_right, green_color, 1)
                    cv2.putText(image, "1", (left_eye_top_left[0], left_eye_bottom_right[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color, 1)
                    cv2.rectangle(image, right_eye_top_left, right_eye_bottom_right, green_color, 1)
                    cv2.putText(image, "1", (right_eye_top_left[0], right_eye_bottom_right[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color, 1)

        cv2.imshow('Blink Detection with Brightness and Sound Control', image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC 키로 종료
            break

capture.release()