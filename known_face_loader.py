# //  ████████      ██          ██          ████████ 
# //  ██            ██          ██          ██       
# //  ██  ████      ██          ██          ██  ████
# //  ██            ██          ██          ██       
# //  ████████      ████████    ████████    ████████ 

import os
import glob
import face_recognition

class KnownFaceLoader:
    def __init__(self, known_faces_root='known_faces'):
        self.known_faces_root = known_faces_root
        self.encodings_dict = {}

    def load_known_faces(self):
        for person_name in os.listdir(self.known_faces_root):
            person_path = os.path.join(self.known_faces_root, person_name)
            if not os.path.isdir(person_path):
                continue
            self.encodings_dict[person_name] = []
            for img_path in glob.glob(os.path.join(person_path, '*.*')):
                image = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(image)
                if encs:
                    self.encodings_dict[person_name].append(encs[0])
                else:
                    print(f"[WARNING] No face found in '{img_path}'")

        print("[INFO] Loaded people:", list(self.encodings_dict.keys()))

    def get_encodings_dict(self):
        return self.encodings_dict
