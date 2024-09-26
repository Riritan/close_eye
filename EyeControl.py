import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.5) as face_mesh:

    blink_counter = 0
    blink_threshold = 15  # 깜빡임으로 간주할 면적 변화 수치

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
                # 눈의 랜드마크 인덱스
                left_eye_indices = [33, 133, 153, 144, 163, 7]
                right_eye_indices = [362, 263, 383, 373, 390, 249]

                # 왼쪽 눈 경계 계산
                left_eye_points = np.array([(face_landmarks.landmark[i].x * image.shape[1],
                                              face_landmarks.landmark[i].y * image.shape[0]) for i in left_eye_indices], np.int32)
                right_eye_points = np.array([(face_landmarks.landmark[i].x * image.shape[1],
                                               face_landmarks.landmark[i].y * image.shape[0]) for i in right_eye_indices], np.int32)

                # 눈 영역의 면적 계산
                left_eye_area = cv2.contourArea(left_eye_points)
                right_eye_area = cv2.contourArea(right_eye_points)

                # 깜빡임 인식
                if left_eye_area < blink_threshold or right_eye_area < blink_threshold:
                    blink_counter += 1
                    cv2.putText(image, "Blink", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    blink_counter = 0  # 깜빡임이 없으면 카운터 초기화

                # 눈 경계 표시
                cv2.polylines(image, [left_eye_points], isClosed=True, color=(0, 255, 0), thickness=1)
                cv2.polylines(image, [right_eye_points], isClosed=True, color=(0, 255, 0), thickness=1)

        cv2.imshow('MediaPipe Face Detection with Blink Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
            break

capture.release()
cv2.destroyAllWindows()