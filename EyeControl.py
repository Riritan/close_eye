import cv2
import mediapipe as mp
import numpy as np
import time 
from close_eye_control_test import setBrightSound

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0)

red_color = (0, 0, 255)
green_color = (0, 255, 0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.5) as face_mesh:

    blink_counter = 0
    blink_threshold = 10  # 깜빡임으로 간주할 면적 변화 수치
    blink_start_time = time.time() #깜빡임 시작 시간
    blink_end_time = 0
    blink_duration =0 # 눈 감긴 시간
    blink_check_time = 3 # 눈 감기는 시간
    is_blinking =False # 감긴 상태 판별
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
                left_eyes = [159, 145]
                right_eyes = [386, 374]

                left_distance = abs(face_landmarks.landmark[left_eyes[0]].y - face_landmarks.landmark[left_eyes[1]].y) * 100
                right_distance = abs(face_landmarks.landmark[right_eyes[0]].y - face_landmarks.landmark[right_eyes[1]].y) * 100

                # left_eye_points = (
                #     face_landmarks.landmark[left_eyes[0]].x - 10,
                #     face_landmarks.landmark[left_eyes[0]].y - 10,
                #     face_landmarks.landmark[left_eyes[1]].x + 10,
                #     face_landmarks.landmark[left_eyes[1]].y + 10,
                # )

                # right_eye_points = [
                #     face_landmarks.landmark[right_eyes[0]].x,
                #     face_landmarks.landmark[right_eyes[0]].y,
                #     face_landmarks.landmark[right_eyes[1]].x,
                #     face_landmarks.landmark[right_eyes[1]].y,
                # ]

                left_eye_top_left = (int(face_landmarks.landmark[left_eyes[0]].x * image.shape[1] - 40),
                                    int(face_landmarks.landmark[left_eyes[0]].y * image.shape[0]) - 20)
                left_eye_bottom_right = (int(face_landmarks.landmark[left_eyes[1]].x * image.shape[1] + 40),
                                        int(face_landmarks.landmark[left_eyes[1]].y * image.shape[0]) + 20)

                right_eye_top_right = (int(face_landmarks.landmark[right_eyes[0]].x * image.shape[1] - 40),
                                    int(face_landmarks.landmark[right_eyes[0]].y * image.shape[0]) - 20)
                right_eye_bottom_right = (int(face_landmarks.landmark[right_eyes[1]].x * image.shape[1] + 40),
                                        int(face_landmarks.landmark[right_eyes[1]].y * image.shape[0]) + 20)

                if (left_distance < 5.5):
                    # 왼쪽 눈이 감겼을 때
                    cv2.rectangle(image, left_eye_top_left, left_eye_bottom_right, red_color, 1)
                    cv2.putText(image, "0", (left_eye_top_left[0], left_eye_bottom_right[1] + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1)

                else:
                    # 왼쪽 눈을 떴을 떄
                    cv2.rectangle(image, left_eye_top_left, left_eye_bottom_right, green_color, 1)
                    cv2.putText(image, "1", (left_eye_top_left[0], left_eye_bottom_right[1] + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color, 1)

                if (right_distance < 5.5):
                    # 오른쪽 눈이 감겼을 때
                    cv2.rectangle(image, right_eye_top_right, right_eye_bottom_right, red_color, 1)
                    cv2.putText(image, "0", (right_eye_top_right[0], right_eye_bottom_right[1] + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1)
                else:
                    # 오른쪽 눈을 떴을 떄
                    cv2.rectangle(image, right_eye_top_right, right_eye_bottom_right, green_color, 1)
                    cv2.putText(image, "1", (right_eye_top_right[0], right_eye_bottom_right[1] + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color, 1)

                if (left_distance < 6 and right_distance < 6):
                    # 눈이 감겼을 때
                    if (not now_blink):
                        blink_start_time = time.time()
                        now_blink = True
                    blink_end_time = time.time()
                    if (blink_end_time - blink_start_time >= blink_check_time and not is_blinking):
                        # 눈 감기는 시간이 지나면 
                        setBrightSound(40, 40)
                        is_blinking = True
                    
                else:
                    # 눈이 떠지면 초기화
                    blink_start_time = time.time()
                    blink_end_time = time.time()
                    is_blinking = False
                    now_blink = False


                # # 눈의 랜드마크 인덱스
                # left_eye_indices = [33, 133, 153, 144, 163, 7]
                # #
                # right_eye_indices = [362, 263, 383, 373, 390, 249]

                # # 왼쪽 눈 경계 계산
                # left_eye_points = np.array([(face_landmarks.landmark[i].x * image.shape[1],
                #                               face_landmarks.landmark[i].y * image.shape[0]) for i in left_eye_indices], np.int32)
                # right_eye_points = np.array([(face_landmarks.landmark[i].x * image.shape[1],
                #                                face_landmarks.landmark[i].y * image.shape[0]) for i in right_eye_indices], np.int32)

                # # 눈 영역의 면적 계산
                # left_eye_area = cv2.contourArea(left_eye_points)
                # right_eye_area = cv2.contourArea(right_eye_points)

                # # 깜빡임 인식
                # if left_eye_area < blink_threshold or right_eye_area < blink_threshold:
                #     if not is_blinking:
                #         is_blinking = True
                #         blink_start_time = time.time()
                #     blink_counter += 1  
                #     cv2.putText(image, "Blink!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # else:
                #     if is_blinking: 
                #         blink_duration = time.time() - blink_start_time
                #         is_blinking = False
                #         cv2.putText(image, f"Blink Duration: {blink_duration:.2f}s",(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                #     # blink_counter = 0  # 깜빡임이 없으면 카운터 초기화

                # # 눈 경계 표시
                # cv2.polylines(image, [left_eye_points], isClosed=True, color=(0, 255, 0), thickness=1)
                # cv2.polylines(image, [right_eye_points], isClosed=True, color=(0, 255, 0), thickness=1)

        cv2.imshow('MediaPipe Face Detection with Blink Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
            break

capture.release()
cv2.destroyAllWindows()
# import cv2
# import mediapipe as mp
# import numpy as np

# mp_face_detection = mp.solutions.face_detection
# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils

# # EAR 계산 함수
# def calculate_EAR(eye_landmarks):
#     A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])  # 세로 길이
#     B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])  # 세로 길이
#     C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])  # 가로 길이
#     EAR = (A + B) / (2.0 * C)
#     return EAR

# capture = cv2.VideoCapture(0)

# with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection, \
#      mp_face_mesh.FaceMesh(min_detection_confidence=0.5) as face_mesh:

#     blink_threshold = 0.2  # EAR 임계값 (눈 감김 인식 기준)
#     blink_counter = 0  # 깜빡임 카운터

#     while capture.isOpened():
#         success, image = capture.read()
#         if not success:
#             print("웹캠을 찾을 수 없습니다")
#             break

#         image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
#         mesh_results = face_mesh.process(image)

#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         if mesh_results.multi_face_landmarks:
#             for face_landmarks in mesh_results.multi_face_landmarks:
#                 # 왼쪽 눈과 오른쪽 눈의 랜드마크 인덱스
#                 left_eye_indices = [33, 133, 153, 144, 163, 7]
#                 right_eye_indices = [362, 263, 383, 373, 390, 249]

#                 # 각 눈의 좌표 추출
#                 left_eye_points = np.array([(face_landmarks.landmark[i].x * image.shape[1],
#                                               face_landmarks.landmark[i].y * image.shape[0]) for i in left_eye_indices], np.float32)
#                 right_eye_points = np.array([(face_landmarks.landmark[i].x * image.shape[1],
#                                                face_landmarks.landmark[i].y * image.shape[0]) for i in right_eye_indices], np.float32)

#                 # EAR 계산
#                 left_EAR = calculate_EAR(left_eye_points)
#                 right_EAR = calculate_EAR(right_eye_points)

#                 # 평균 EAR
#                 avg_EAR = (left_EAR + right_EAR) / 2.0

#                 # 깜빡임 인식
#                 if avg_EAR < blink_threshold:
#                     blink_counter += 1
#                     cv2.putText(image, "Blink", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                 else:
#                     blink_counter = 0  # 깜빡임이 없을 경우 카운터 초기화

#                 # 눈 경계 표시
#                 cv2.polylines(image, [left_eye_points.astype(int)], isClosed=True, color=(0, 255, 0), thickness=1)
#                 cv2.polylines(image, [right_eye_points.astype(int)], isClosed=True, color=(0, 255, 0), thickness=1)

#                 # EAR 값 화면에 표시
#                 cv2.putText(image, f'EAR: {avg_EAR:.2f}', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#         cv2.imshow('MediaPipe Face Detection with EAR Blink Detection', image)
#         if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
#             break

# capture.release()
# cv2.destroyAllWindows()