# //  ████████      ██          ██          ████████ 
# //  ██            ██          ██          ██       
# //  ██  ████      ██          ██          ██  ████
# //  ██            ██          ██          ██       
# //  ████████      ████████    ████████    ████████ 

import cv2
import face_recognition
import numpy as np

class FaceRecognitionService:
    def __init__(self, encodings_dict, tolerance=0.6, camera_index=0, skip_frames=2, scale=0.5):
        self.encodings_dict = encodings_dict
        self.tolerance = tolerance
        self.camera_index = camera_index
        self.skip_frames = skip_frames
        self.scale = scale
        self.frame_count = 0
        self.video_capture = None

    def start(self):
        self.video_capture = cv2.VideoCapture(self.camera_index)
        if not self.video_capture.isOpened():
            print(f"[ERROR] Cannot open webcam at index {self.camera_index}.")
            return

        print("[INFO] Face recognition started. Press 'q' to quit.")
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                print("[ERROR] Failed to read frame.")
                break

            self.frame_count += 1
            cv2.imshow("Live View", frame)

            if self.frame_count % self.skip_frames != 0:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            small_frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb_small_frame)
            encodings = face_recognition.face_encodings(rgb_small_frame, locations)

            for (top, right, bottom, left), unknown_enc in zip(locations, encodings):
                name = self._recognize_face(unknown_enc)
                s = 1 / self.scale
                top, right, bottom, left = int(top*s), int(right*s), int(bottom*s), int(left*s)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Processed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop()

    def _recognize_face(self, unknown_enc):
        best_match = "Unknown"
        best_distance = float("inf")

        for person_name, person_encodings in self.encodings_dict.items():
            matches = face_recognition.compare_faces(person_encodings, unknown_enc, self.tolerance)
            dist = face_recognition.face_distance(person_encodings, unknown_enc)
            if len(dist) > 0:
                min_dist = np.min(dist)
                if min_dist < best_distance and any(matches):
                    best_distance = min_dist
                    best_match = person_name
        return best_match

    def stop(self):
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()
        cv2.destroyAllWindows()
