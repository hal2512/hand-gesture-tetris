import cv2 as cv
import numpy as np
import copy
import csv
import itertools
import mediapipe as mp
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier


class GestureDetector:
    def __init__(self, cam_index=0, width=640, height=480):
        self.cap = cv.VideoCapture(cam_index)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        self.keypoint_classifier = KeyPointClassifier()
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            self.labels = [row[0] for row in reader]

        self.current_label = "None"

    def get_current_gesture(self):
        ret, frame = self.cap.read()
        if not ret:
            return "None"

        frame = cv.flip(frame, 1)
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = self._calc_landmark_list(frame, hand_landmarks)
                preprocessed = self._preprocess_landmarks(landmark_list)
                gesture_id = self.keypoint_classifier(preprocessed)
                self.current_label = self.labels[gesture_id]
                return self.current_label

        return "None"

    def _calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        for landmark in landmarks.landmark:
            x = min(int(landmark.x * image_width), image_width - 1)
            y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([x, y])
        return landmark_point

    def _preprocess_landmarks(self, landmark_list):
        temp = copy.deepcopy(landmark_list)
        base_x, base_y = temp[0][0], temp[0][1]
        for i in range(len(temp)):
            temp[i][0] -= base_x
            temp[i][1] -= base_y
        temp = list(itertools.chain.from_iterable(temp))
        max_value = max(list(map(abs, temp)))
        return [v / max_value for v in temp] if max_value != 0 else temp

    def release(self):
        self.cap.release()
        cv.destroyAllWindows()
