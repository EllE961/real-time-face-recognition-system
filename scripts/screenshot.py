"""Screenshot both tabs of the real app, with a photo standing in for the webcam.

known_faces/ is pointed at docs/demo/known_faces (NASA portraits of the Artemis II crew),
cv2.VideoCapture returns docs/demo/artemis2_crew_frame.jpg (a different NASA photo of the
same four people), and the "Encodings Loaded" message box is skipped. The window is created
with Qt.WA_DontShowOnScreen, so nothing appears on the desktop; QWidget.grab() saves it.

    python scripts/screenshot.py
"""
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DEMO = os.path.join(ROOT, "docs", "demo")
FRAME = os.path.join(DEMO, "artemis2_crew_frame.jpg")
OUT = os.path.join(ROOT, "docs", "assets")


class StillCapture:
    def __init__(self, *_):
        self.frame = cv2.imread(FRAME)

    def isOpened(self):
        return self.frame is not None

    def read(self):
        return True, self.frame.copy()

    def release(self):
        pass


cv2.VideoCapture = StillCapture

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402

import face_recognizer as fr  # noqa: E402

fr.KNOWN_FACES_DIR = os.path.join(DEMO, "known_faces")
QMessageBox.information = staticmethod(lambda *a, **k: None)

app = QApplication(sys.argv)
win = fr.MainWindow()
win.setAttribute(Qt.WA_DontShowOnScreen)
win.show()
app.processEvents()

# Face Manager tab with one person and one image selected
mgr = win.tab_manager
mgr.load_people()
mgr.people_list.setCurrentRow([mgr.people_list.item(i).text() for i in range(mgr.people_list.count())].index("Christina Koch"))
app.processEvents()
mgr.images_list.setCurrentRow(0)
app.processEvents()
win.grab().save(os.path.join(OUT, "face_manager.png"))

# Recognition tab: load encodings, "open the camera", then run frames by hand instead of the timer
rec = win.tab_recognition
win.tab_widget.setCurrentWidget(rec)
app.processEvents()
rec.start_recognition()
rec.timer.stop()
for _ in range(8):  # a few frames so the video label settles at its final size
    rec.process_frame()
    app.processEvents()
win.grab().save(os.path.join(OUT, "recognition.png"))
print("encodings per person:", {k: len(v) for k, v in rec.encodings_dict.items()})
print("faces found:", [(b[4], b[:4]) for b in rec.last_boxes])
