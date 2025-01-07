#  ████████      ██          ██          ████████ 
#  ██            ██          ██          ██       
#  ██  ████      ██          ██          ██  ████
#  ██            ██          ██          ██       
#  ████████      ████████    ████████    ████████ 

import os
import glob
import face_recognition

class KnownFace:
    # Stores a name and face encoding
    def __init__(self, name, encoding):
        self.name = name
        self.encoding = encoding

class KnownFaceLoader:
    # Loads images from a folder and extracts face encodings
    def __init__(self, known_faces_folder='known_faces'):
        self.known_faces_folder = known_faces_folder
        self.known_faces = []

    def load_known_faces(self):
        image_paths = glob.glob(os.path.join(self.known_faces_folder, '*.*'))
        for image_path in image_paths:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                face_encoding = encodings[0]
                base_name = os.path.basename(image_path)
                name = os.path.splitext(base_name)[0]
                self.known_faces.append(KnownFace(name, face_encoding))
            else:
                print(f"[WARNING] No face found in '{image_path}'. Skipping.")

        loaded_names = [face.name for face in self.known_faces]
        print("[INFO] Loaded known faces:", loaded_names)
