#  ████████      ██          ██          ████████ 
#  ██            ██          ██          ██       
#  ██  ████      ██          ██          ██  ████
#  ██            ██          ██          ██       
#  ████████      ████████    ████████    ████████ 

import cv2
import face_recognition
import numpy as np
from known_face_loader import KnownFace

class FaceRecognitionService:
    # Handles real-time face recognition from webcam
    def __init__(self, known_faces, tolerance=0.6, camera_index=0):
        self.known_faces = known_faces
        self.tolerance = tolerance
        self.camera_index = camera_index
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

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                known_encodings = [f.encoding for f in self.known_faces]
                known_names = [f.name for f in self.known_faces]

                matches = face_recognition.compare_faces(known_encodings, face_encoding, self.tolerance)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

                if best_match_index is not None and matches[best_match_index]:
                    name = known_names[best_match_index]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, bottom + 25),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Exiting recognition...")
                break

        self.stop()

    def stop(self):
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()
        cv2.destroyAllWindows()
