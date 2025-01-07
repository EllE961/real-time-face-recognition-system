# //  ████████      ██          ██          ████████ 
# //  ██            ██          ██          ██       
# //  ██  ████      ██          ██          ██  ████
# //  ██            ██          ██          ██       
# //  ████████      ████████    ████████    ████████ 

from known_face_loader import KnownFaceLoader
from face_recognition_service import FaceRecognitionService

def main():
    loader = KnownFaceLoader('known_faces')
    loader.load_known_faces()
    encodings_dict = loader.get_encodings_dict()

    service = FaceRecognitionService(
        encodings_dict=encodings_dict,
        tolerance=0.6,
        camera_index=0,
        skip_frames=2,
        scale=0.5
    )
    service.start()

if __name__ == '__main__':
    main()
