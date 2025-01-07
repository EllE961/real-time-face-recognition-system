# //  ████████      ██          ██          ████████
# //  ██            ██          ██          ██
# //  ██  ████      ██          ██          ██  ████
# //  ██            ██          ██          ██
# //  ████████      ████████    ████████    ████████

import os
import sys
import cv2
import glob
import shutil
import face_recognition
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QDir
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QLabel,
    QLineEdit, QMessageBox, QFileDialog,
    QSpinBox, QDoubleSpinBox, QFormLayout
)

KNOWN_FACES_DIR = "known_faces"

class FaceManagerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QHBoxLayout())

        # Lists and preview area
        self.people_list = QListWidget()
        self.images_list = QListWidget()
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)

        # Buttons and input
        self.add_person_btn = QPushButton("Add Person")
        self.del_person_btn = QPushButton("Delete Person")
        self.add_image_btn = QPushButton("Add Image")
        self.del_image_btn = QPushButton("Delete Image")
        self.capture_btn = QPushButton("Capture Webcam")
        self.person_input = QLineEdit()
        self.person_input.setPlaceholderText("Person Name")

        # Layout: left (people), right (images), preview (center)
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.people_list)
        left_layout.addWidget(self.person_input)
        left_layout.addWidget(self.add_person_btn)
        left_layout.addWidget(self.del_person_btn)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.images_list)
        right_layout.addWidget(self.add_image_btn)
        right_layout.addWidget(self.del_image_btn)
        right_layout.addWidget(self.capture_btn)

        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.preview_label)

        self.layout().addLayout(left_layout, 2)
        self.layout().addLayout(right_layout, 2)
        self.layout().addLayout(preview_layout, 3)

        self.people_list.currentTextChanged.connect(self.on_person_selected)
        self.images_list.currentTextChanged.connect(self.on_image_selected)
        self.add_person_btn.clicked.connect(self.on_add_person)
        self.del_person_btn.clicked.connect(self.on_del_person)
        self.add_image_btn.clicked.connect(self.on_add_image)
        self.del_image_btn.clicked.connect(self.on_del_image)
        self.capture_btn.clicked.connect(self.on_capture_image)

        self.load_people()

    def load_people(self):
        self.people_list.clear()
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR)
        for name in os.listdir(KNOWN_FACES_DIR):
            person_dir = os.path.join(KNOWN_FACES_DIR, name)
            if os.path.isdir(person_dir):
                self.people_list.addItem(name)

    def on_person_selected(self, person_name):
        self.images_list.clear()
        self.preview_label.clear()
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if os.path.isdir(person_dir):
            for f in os.listdir(person_dir):
                self.images_list.addItem(f)

    def on_image_selected(self, img_name):
        item = self.people_list.currentItem()
        if not item:
            return
        person_name = item.text()
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        img_path = os.path.join(person_dir, img_name)
        if os.path.isfile(img_path):
            pix = QPixmap(img_path)
            if not pix.isNull():
                scaled = pix.scaled(
                    self.preview_label.width(),
                    self.preview_label.height(),
                    Qt.KeepAspectRatio
                )
                self.preview_label.setPixmap(scaled)
            else:
                self.preview_label.setText("Cannot preview this file.")
        else:
            self.preview_label.setText("")

    def on_add_person(self):
        name = self.person_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Enter a person name.")
            return
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if os.path.exists(person_dir):
            QMessageBox.warning(self, "Warning", "Person already exists.")
            return
        os.makedirs(person_dir)
        self.load_people()
        self.person_input.clear()

    def on_del_person(self):
        item = self.people_list.currentItem()
        if not item:
            return
        person_name = item.text()
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        confirm = QMessageBox.question(
            self, "Delete Person",
            f"Delete '{person_name}' and all images?",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            shutil.rmtree(person_dir)
            self.load_people()

    def on_add_image(self):
        person_item = self.people_list.currentItem()
        if not person_item:
            QMessageBox.warning(self, "Warning", "Select a person first.")
            return
        person_name = person_item.text()
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", QDir.homePath(),
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if not file_paths:
            return
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        for src_path in file_paths:
            base_name = os.path.basename(src_path)
            dest_path = os.path.join(person_dir, base_name)
            shutil.copyfile(src_path, dest_path)
        self.on_person_selected(person_name)

    def on_del_image(self):
        person_item = self.people_list.currentItem()
        if not person_item:
            return
        person_name = person_item.text()
        image_item = self.images_list.currentItem()
        if not image_item:
            return
        img_name = image_item.text()
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        img_path = os.path.join(person_dir, img_name)
        confirm = QMessageBox.question(
            self, "Delete Image",
            f"Delete '{img_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm == QMessageBox.Yes and os.path.isfile(img_path):
            os.remove(img_path)
            self.on_person_selected(person_name)

    def on_capture_image(self):
        person_item = self.people_list.currentItem()
        if not person_item:
            QMessageBox.warning(self, "Warning", "Select a person first.")
            return
        person_name = person_item.text()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open webcam.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                QMessageBox.critical(self, "Error", "Failed to read webcam.")
                break
            cv2.imshow("Capture: Space to save, ESC to cancel", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == 32:
                person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
                img_name = f"captured_{len(os.listdir(person_dir))}.jpg"
                cv2.imwrite(os.path.join(person_dir, img_name), frame)
                break
        cap.release()
        cv2.destroyAllWindows()
        self.on_person_selected(person_name)

class RecognitionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.encodings_dict = {}
        self.running = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.cap = None
        self.frame_count = 0
        self.last_boxes = []

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)

        # Buttons and input
        self.btn_load_encodings = QPushButton("Reload Encodings")
        self.btn_start = QPushButton("Start Recognition")
        self.btn_stop = QPushButton("Stop Recognition")

        self.spin_skip = QSpinBox()
        self.spin_skip.setRange(1, 30)
        self.spin_skip.setValue(2)

        self.dspin_scale = QDoubleSpinBox()
        self.dspin_scale.setRange(0.1, 1.0)
        self.dspin_scale.setSingleStep(0.1)
        self.dspin_scale.setValue(0.5)

        self.dspin_tolerance = QDoubleSpinBox()
        self.dspin_tolerance.setRange(0.1, 1.0)
        self.dspin_tolerance.setSingleStep(0.05)
        self.dspin_tolerance.setValue(0.6)

        # Layout
        form_layout = QFormLayout()
        form_layout.addRow("Skip Frames:", self.spin_skip)
        form_layout.addRow("Scale Factor:", self.dspin_scale)
        form_layout.addRow("Tolerance:", self.dspin_tolerance)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_load_encodings)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)

        self.layout().addLayout(form_layout)
        self.layout().addLayout(btn_layout)
        self.layout().addWidget(self.video_label)

        self.btn_load_encodings.clicked.connect(self.load_encodings)
        self.btn_start.clicked.connect(self.start_recognition)
        self.btn_stop.clicked.connect(self.stop_recognition)

    def load_encodings(self):
        self.encodings_dict.clear()
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR)
        for person_name in os.listdir(KNOWN_FACES_DIR):
            person_path = os.path.join(KNOWN_FACES_DIR, person_name)
            if not os.path.isdir(person_path):
                continue
            self.encodings_dict[person_name] = []
            for img_path in glob.glob(os.path.join(person_path, '*.*')):
                image = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(image)
                if encs:
                    self.encodings_dict[person_name].append(encs[0])
        QMessageBox.information(self, "Encodings Loaded", "Encodings have been reloaded.")

    def start_recognition(self):
        if self.running:
            return
        self.load_encodings()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open webcam.")
            return
        self.running = True
        self.frame_count = 0
        self.last_boxes = []
        self.timer.start(30)

    def stop_recognition(self):
        self.running = False
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.video_label.clear()
        self.last_boxes = []

    def process_frame(self):
        if not self.running or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        self.frame_count += 1
        skip_frames = self.spin_skip.value()
        if (self.frame_count % skip_frames) != 0:
            self.draw_boxes(frame, self.last_boxes)
            self.display_frame(frame)
            return
        scale = self.dspin_scale.value()
        tolerance = self.dspin_tolerance.value()
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb_small_frame)
        encs = face_recognition.face_encodings(rgb_small_frame, locs)
        new_boxes = []
        for (top, right, bottom, left), unknown_enc in zip(locs, encs):
            name = self._recognize(unknown_enc, tolerance)
            s = 1 / scale
            top, right, bottom, left = int(top*s), int(right*s), int(bottom*s), int(left*s)
            new_boxes.append((left, top, right, bottom, name))
        self.last_boxes = new_boxes
        self.draw_boxes(frame, self.last_boxes)
        self.display_frame(frame)

    def draw_boxes(self, frame, boxes):
        for (left, top, right, bottom, name) in boxes:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    def _recognize(self, unknown_enc, tolerance):
        best_name = "Unknown"
        best_distance = float("inf")
        for person_name, enc_list in self.encodings_dict.items():
            matches = face_recognition.compare_faces(enc_list, unknown_enc, tolerance)
            dist = face_recognition.face_distance(enc_list, unknown_enc)
            if len(dist) > 0:
                min_dist = np.min(dist)
                if min_dist < best_distance and any(matches):
                    best_distance = min_dist
                    best_name = person_name
        return best_name

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognizer")
        self.setGeometry(50, 50, 1100, 700)
        self.tab_widget = QTabWidget()
        self.tab_manager = FaceManagerTab()
        self.tab_recognition = RecognitionTab()
        self.tab_widget.addTab(self.tab_manager, "Face Manager")
        self.tab_widget.addTab(self.tab_recognition, "Recognition")
        self.setCentralWidget(self.tab_widget)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
