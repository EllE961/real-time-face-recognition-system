#  ████████      ██          ██          ████████ 
#  ██            ██          ██          ██       
#  ██  ████      ██          ██          ██  ████
#  ██            ██          ██          ██       
#  ████████      ████████    ████████    ████████ 

from known_face_loader import KnownFaceLoader
from face_recognition_service import FaceRecognitionService

def main():
    loader = KnownFaceLoader('known_faces')
    loader.load_known_faces()

    service = FaceRecognitionService(
        known_faces=loader.known_faces,
        tolerance=0.6,
        camera_index=0
    )

    # Start face recognition
    service.start()

if __name__ == '__main__':
    main()
