import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
import mediapipe as mp

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QTextEdit, QDialog, QHBoxLayout, QSpinBox, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ===================== Model Initialization =====================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()
COCO_INSTANCE_CATEGORY_NAMES = weights.meta["categories"]

# ===================== Calibration & Constants =====================
KNOWN_HEIGHT_CM   = 173
CALIBRATION_TOP   = (315, 70)
CALIBRATION_BOTTOM= (320, 345)
ASSUMED_BMI       = 22
CONFIDENCE_THRES  = 0.6
OUTLINE_COLOR     = (0, 255, 255)
OUTLINE_THICKNESS = 3
FILL_ALPHA        = 0.25

PIXEL_PER_CM = abs(CALIBRATION_BOTTOM[1] - CALIBRATION_TOP[1]) / KNOWN_HEIGHT_CM

def calibrate_pixel_per_cm(top, bottom, known_height_cm):
    return abs(bottom[1] - top[1]) / known_height_cm

# ===================== Helper Functions =====================

def smooth_mask(mask):
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask

def draw_silhouette(frame: np.ndarray, mask: np.ndarray):
    h, w = frame.shape[:2]
    mask = smooth_mask(mask)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return
    cnt = max(contours, key=cv2.contourArea)
    epsilon = 0.003 * cv2.arcLength(cnt, True)
    smooth = cv2.approxPolyDP(cnt, epsilon, True)
    if FILL_ALPHA > 0:
        overlay = frame.copy()
        cv2.drawContours(overlay, [smooth], -1, OUTLINE_COLOR, thickness=-1, lineType=cv2.LINE_AA)
        frame[:] = cv2.addWeighted(overlay, FILL_ALPHA, frame, 1 - FILL_ALPHA, 0)
    cv2.polylines(frame, [smooth], isClosed=True, color=OUTLINE_COLOR, thickness=OUTLINE_THICKNESS, lineType=cv2.LINE_AA)

def moving_average(data, window_size=5):
    if len(data) < window_size:
        return np.mean(data)
    return np.mean(data[-window_size:])

# ===================== Calibration Wizard =====================

class CalibrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Wizard")
        self.top_y = QSpinBox()
        self.top_y.setRange(0, 1000)
        self.top_y.setValue(CALIBRATION_TOP[1])
        self.bottom_y = QSpinBox()
        self.bottom_y.setRange(0, 1000)
        self.bottom_y.setValue(CALIBRATION_BOTTOM[1])
        self.height_cm = QSpinBox()
        self.height_cm.setRange(50, 250)
        self.height_cm.setValue(KNOWN_HEIGHT_CM)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Top Y:"))
        layout.addWidget(self.top_y)
        layout.addWidget(QLabel("Bottom Y:"))
        layout.addWidget(self.bottom_y)
        layout.addWidget(QLabel("Known Height (cm):"))
        layout.addWidget(self.height_cm)
        btn = QPushButton("Apply")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)
        self.setLayout(layout)

    def get_values(self):
        return (self.top_y.value(), self.bottom_y.value(), self.height_cm.value())

# ===================== Camera Worker Thread =====================

class CameraThread(QThread):
    frame_updated = pyqtSignal(np.ndarray, str)

    def __init__(self, cam_idx: int = 0):
        super().__init__()
        self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        self.running = self.cap.isOpened()
        self.height_history = []
        self.pixel_per_cm = PIXEL_PER_CM

    def set_calibration(self, top, bottom, known_height):
        self.pixel_per_cm = calibrate_pixel_per_cm((0, top), (0, bottom), known_height)

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                break
            report = self.process_frame(frame)
            self.frame_updated.emit(frame, report)

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.quit()

    def process_frame(self, frame: np.ndarray) -> str:
        report_lines = []
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(rgb)
        cm_height = None

        if pose_res.pose_landmarks:
            landmarks = pose_res.pose_landmarks.landmark
            y_coords = [landmarks[mp_pose.PoseLandmark.NOSE].y,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
            top_y = min(y_coords) * h
            bottom_y = max(y_coords) * h
            px_height = bottom_y - top_y
            cm_height = px_height / self.pixel_per_cm
            self.height_history.append(cm_height)
            avg_height = moving_average(self.height_history)
            est_weight = ASSUMED_BMI * (avg_height / 100) ** 2

            if pose_res.segmentation_mask is not None:
                mask = (pose_res.segmentation_mask.squeeze() > 0.1).astype("uint8") * 255
                mask = cv2.resize(mask, (w, h))
                draw_silhouette(frame, mask)

            mp_draw.draw_landmarks(
                frame,
                pose_res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(180, 180, 180), thickness=1),
            )

            report_lines.append(f"🧍 Height≈{avg_height:.1f} cm  Weight≈{est_weight:.1f} kg")

        # Object detection
        with torch.no_grad():
            preds = model(F.to_tensor(rgb).unsqueeze(0))[0]

        detected_objects_count = 0  # Counter for detected objects

        for box, score, lbl in zip(preds["boxes"], preds["scores"], preds["labels"]):
            if score < CONFIDENCE_THRES:
                continue
            detected_objects_count += 1

            x1, y1, x2, y2 = box.int().tolist()
            width_cm  = (x2 - x1) / self.pixel_per_cm
            height_cm = (y2 - y1) / self.pixel_per_cm
            volume_cm3 = width_cm * height_cm * 5.0
            label = COCO_INSTANCE_CATEGORY_NAMES[lbl] if lbl < len(COCO_INSTANCE_CATEGORY_NAMES) else f"id:{lbl}"
            caption = f"{label}  {width_cm:.1f}×{height_cm:.1f} cm  ({volume_cm3:.0f} cm³)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, caption, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 255, 255), 2, lineType=cv2.LINE_AA)
            report_lines.append(
                f"📦 {label}: W={width_cm:.1f} cm  H={height_cm:.1f} cm  V={volume_cm3:.0f} cm³"
            )

        # Add total object count to report
        report_lines.append(f"🔢 Total objects detected: {detected_objects_count}")

        return "\n".join(report_lines)

# ===================== PyQt5 GUI =====================

class EstimatorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📏 Human & Object Estimator")
        self.setStyleSheet("background:#121212;color:#ddd;")
        self.resize(920, 720)

        self.lbl_view = QLabel(alignment=Qt.AlignCenter)
        self.txt_report = QTextEdit(readOnly=True)
        self.txt_report.setStyleSheet("background:#1e1e1e;color:#00ffaa;font-size:14px;")

        self.btn_start = QPushButton("▶ Start Camera")
        self.btn_start.setStyleSheet("padding:10px 20px;background:#007acc;border:none;color:#fff;")
        self.btn_start.clicked.connect(self.start_cam)

        self.btn_calibrate = QPushButton("🛠 Calibrate")
        self.btn_calibrate.setStyleSheet("padding:10px 20px;background:#ffaa00;border:none;color:#222;")
        self.btn_calibrate.clicked.connect(self.calibrate)

        lay = QVBoxLayout(self)
        lay.addWidget(self.lbl_view)
        lay.addWidget(self.txt_report)
        h_lay = QHBoxLayout()
        h_lay.addWidget(self.btn_start)
        h_lay.addWidget(self.btn_calibrate)
        lay.addLayout(h_lay)

        self.cam_thread: CameraThread | None = None

    def start_cam(self):
        if self.cam_thread is None:
            self.cam_thread = CameraThread(0)
            self.cam_thread.frame_updated.connect(self.update_gui)
            self.cam_thread.start()
            self.btn_start.setDisabled(True)

    def calibrate(self):
        dlg = CalibrationDialog(self)
        if dlg.exec():
            top, bottom, height_cm = dlg.get_values()
            if self.cam_thread:
                self.cam_thread.set_calibration(top, bottom, height_cm)

    def update_gui(self, frame: np.ndarray, report: str):
        h, w, _ = frame.shape
        qimg = QImage(frame.data, w, h, frame.strides[0], QImage.Format_BGR888)
        self.lbl_view.setPixmap(QPixmap.fromImage(qimg))
        self.txt_report.setPlainText(report)

    def closeEvent(self, ev):
        if self.cam_thread:
            self.cam_thread.stop()
        ev.accept()

# ===================== main =====================
if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication([])
    win = EstimatorApp()
    win.show()
    app.exec()
