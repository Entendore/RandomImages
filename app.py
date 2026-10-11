"""
Pixel Randomizer — Multi-Image Morph App with MP4 Export
Built with PySide6, Pillow, NumPy, and imageio.
"""

import sys
import io
import urllib.request
import urllib.error
import numpy as np
from PIL import Image

try:
    import imageio
except ImportError:
    imageio = None

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QLineEdit, QSlider,
    QGroupBox, QScrollArea, QMessageBox, QProgressBar, QComboBox,
    QCheckBox, QSpinBox, QDialog, QDialogButtonBox, QFormLayout,
    QDoubleSpinBox
)
from PySide6.QtCore import QTimer, Qt, QSize, Signal, QThread
from PySide6.QtGui import QPixmap, QImage, QFont


# ──────────────────────────── Helpers ────────────────────────────

def pil_to_qpixmap(pil_image: Image.Image) -> QPixmap:
    rgb = pil_image.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    qimg = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

def load_image_from_url(url: str) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": "PixelRandomizer/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = resp.read()
    return Image.open(io.BytesIO(data))


# ──────────────────────────── Frame Generator ────────────────────────────

def generate_morph_frames(image_arrays: list[np.ndarray], fps: int, seconds_per_transition: float):
    """
    Yields (frame, progress_pct) tuples for the full morph sequence.
    Each frame is an (H, W, 3) uint8 numpy array.
    """
    if len(image_arrays) < 2:
        return

    h, w, _ = image_arrays[0].shape
    total_pixels = h * w
    frames_per_transition = max(1, int(fps * seconds_per_transition))
    pixels_per_frame = max(1, int(np.ceil(total_pixels / frames_per_transition)))

    total_transitions = len(image_arrays) - 1
    total_frames = total_transitions * frames_per_transition
    frame_counter = 0

    current = image_arrays[0].copy()
    yield current.copy(), 0

    for i in range(total_transitions):
        target = image_arrays[i + 1]
        indices = np.arange(total_pixels, dtype=np.int32)
        np.random.shuffle(indices)
        cursor = 0

        while cursor < total_pixels:
            batch_end = min(cursor + pixels_per_frame, total_pixels)
            flat_idx = indices[cursor:batch_end]
            rows = flat_idx // w
            cols = flat_idx % w
            current[rows, cols] = target[rows, cols]
            cursor = batch_end

            frame_counter += 1
            pct = int(100 * frame_counter / total_frames)
            yield current.copy(), min(pct, 100)


# ──────────────────────────── Export Worker ────────────────────────────

class ExportWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished_ok = Signal(str)
    finished_err = Signal(str)

    def __init__(self, image_arrays, output_path, fps, seconds_per_transition, parent=None):
        super().__init__(parent)
        self.image_arrays = image_arrays
        self.output_path = output_path
        self.fps = fps
        self.seconds_per_transition = seconds_per_transition
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            if imageio is None:
                self.finished_err.emit("imageio is not installed. Run: pip install 'imageio[ffmpeg]'")
                return

            self.status.emit("Initializing video writer…")
            h, w, _ = self.image_arrays[0].shape

            writer = imageio.get_writer(
                self.output_path,
                fps=self.fps,
                codec="libx264",
                quality=8,
                macro_block_size=2,   # allow odd resolutions
                output_params=["-pix_fmt", "yuv420p"],  # compatibility
            )

            try:
                gen = generate_morph_frames(
                    self.image_arrays, self.fps, self.seconds_per_transition
                )
                for frame, pct in gen:
                    if self._cancel:
                        self.status.emit("Cancelled.")
                        writer.close()
                        return
                    writer.append_data(frame)
                    self.progress.emit(pct)
            finally:
                writer.close()

            self.finished_ok.emit(self.output_path)
        except Exception as e:
            self.finished_err.emit(f"Export failed:\n{e}")


# ──────────────────────────── Export Dialog ────────────────────────────

class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export MP4")
        self.setMinimumWidth(320)
        self._build_ui()

    def _build_ui(self):
        layout = QFormLayout(self)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)
        self.fps_spin.setSuffix(" fps")
        layout.addRow("Frame rate:", self.fps_spin)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.2, 60.0)
        self.duration_spin.setValue(2.0)
        self.duration_spin.setSingleStep(0.5)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setDecimals(1)
        layout.addRow("Duration per transition:", self.duration_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def values(self):
        return self.fps_spin.value(), self.duration_spin.value()


# ──────────────────────────── Image Slot ────────────────────────────

class ImageSlot(QGroupBox):
    image_changed = Signal()
    remove_requested = Signal(object)

    def __init__(self, index: int, parent=None):
        super().__init__(f"Image {index}", parent)
        self.image: Image.Image | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        input_row = QHBoxLayout()
        self.btn_load_file = QPushButton("📁 File")
        self.btn_load_file.clicked.connect(self._load_file)
        input_row.addWidget(self.btn_load_file)

        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("Or paste image URL…")
        input_row.addWidget(self.url_edit)

        self.btn_load_url = QPushButton("🌐 URL")
        self.btn_load_url.clicked.connect(self._load_url)
        input_row.addWidget(self.btn_load_url)

        self.btn_remove = QPushButton("✕")
        self.btn_remove.setFixedWidth(35)
        self.btn_remove.setStyleSheet("QPushButton { color: #f55; font-weight: bold; }")
        self.btn_remove.clicked.connect(lambda: self.remove_requested.emit(self))
        input_row.addWidget(self.btn_remove)

        layout.addLayout(input_row)

        self.preview = QLabel("No image")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setFixedHeight(120)
        self.preview.setStyleSheet(
            "QLabel { border: 1px dashed #555; border-radius: 4px; background: #1a1a1a; color: #666; }"
        )
        layout.addWidget(self.preview)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #8f8; font-size: 11px;")
        layout.addWidget(self.status_label)

    def _set_image(self, img: Image.Image, source: str):
        self.image = img
        pm = pil_to_qpixmap(img)
        scaled = pm.scaled(200, 116, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.preview.setPixmap(scaled)
        self.status_label.setText(f"✔ Loaded ({img.width}×{img.height}) from {source}")
        self.image_changed.emit()

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.gif *.tiff);;All Files (*)"
        )
        if not path: return
        try:
            img = Image.open(path)
            img.load()
            self._set_image(img, "file")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load:\n{e}")

    def _load_url(self):
        url = self.url_edit.text().strip()
        if not url: return
        self.status_label.setText("⏳ Downloading…")
        QApplication.processEvents()
        try:
            img = load_image_from_url(url)
            img.load()
            self._set_image(img, "URL")
        except Exception as e:
            self.status_label.setText("")
            QMessageBox.warning(self, "Error", f"Failed to download:\n{e}")


# ──────────────────────────── Main Window ────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixel Randomizer — Multi-Image Morph + MP4 Export")
        self.resize(1000, 850)

        self.slots: list[ImageSlot] = []
        self.image_arrays: list[np.ndarray] = []
        self.current_arr: np.ndarray | None = None
        self.img_a_arr: np.ndarray | None = None
        self.img_b_arr: np.ndarray | None = None
        self.shuffled_indices: np.ndarray | None = None
        self.cursor: int = 0
        self.total_pixels: int = 0
        self.current_transition_idx: int = 0
        self.animating: bool = False
        self.export_worker: ExportWorker | None = None

        self._build_ui()
        self._setup_timer()

        self._add_slot()
        self._add_slot()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- Image Queue ---
        queue_group = QGroupBox("Image Queue")
        queue_layout = QVBoxLayout(queue_group)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("➕ Add Image")
        self.btn_add.clicked.connect(self._add_slot)
        btn_row.addWidget(self.btn_add)

        self.btn_clear = QPushButton("🗑️ Clear All")
        self.btn_clear.clicked.connect(self._clear_all)
        btn_row.addWidget(self.btn_clear)
        btn_row.addStretch()
        queue_layout.addLayout(btn_row)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(200)
        self.scroll_area.setMaximumHeight(350)

        self.scroll_widget = QWidget()
        self.slot_layout = QVBoxLayout(self.scroll_widget)
        self.slot_layout.addStretch()

        self.scroll_area.setWidget(self.scroll_widget)
        queue_layout.addWidget(self.scroll_area)

        main_layout.addWidget(queue_group)

        # --- Controls ---
        ctrl_group = QGroupBox("Playback Controls")
        ctrl_layout = QHBoxLayout(ctrl_group)

        self.btn_start = QPushButton("▶  Start Morph")
        self.btn_start.clicked.connect(self._start_morph)
        ctrl_layout.addWidget(self.btn_start)

        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_pause.setEnabled(False)
        ctrl_layout.addWidget(self.btn_pause)

        self.btn_reset = QPushButton("↺  Reset")
        self.btn_reset.clicked.connect(self._reset)
        ctrl_layout.addWidget(self.btn_reset)

        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(40)
        self.speed_slider.setFixedWidth(150)
        ctrl_layout.addWidget(self.speed_slider)

        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(QLabel("Res:"))
        self.res_combo = QComboBox()
        self.res_combo.addItems(["128", "256", "512", "Original"])
        self.res_combo.setCurrentIndex(1)
        ctrl_layout.addWidget(self.res_combo)

        self.loop_check = QCheckBox("Loop")
        ctrl_layout.addWidget(self.loop_check)

        main_layout.addWidget(ctrl_group)

        # --- Export ---
        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout(export_group)

        self.btn_export = QPushButton("💾  Export to MP4…")
        self.btn_export.clicked.connect(self._export_mp4)
        export_layout.addWidget(self.btn_export)

        self.btn_cancel_export = QPushButton("✕ Cancel Export")
        self.btn_cancel_export.clicked.connect(self._cancel_export)
        self.btn_cancel_export.setEnabled(False)
        export_layout.addWidget(self.btn_cancel_export)

        self.export_status = QLabel("")
        self.export_status.setStyleSheet("color: #8cf; font-size: 12px;")
        export_layout.addWidget(self.export_status, stretch=1)

        main_layout.addWidget(export_group)

        # --- Progress ---
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        main_layout.addWidget(self.progress)

        # --- Canvas ---
        self.canvas = QLabel("Add images and press Start")
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas.setStyleSheet(
            "QLabel { background: #111; border: 2px solid #444; border-radius: 8px; color: #666; font-size: 16px; }"
        )
        self.canvas.setMinimumSize(400, 300)
        main_layout.addWidget(self.canvas, stretch=1)

        self._check_ready()

    def _setup_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    # ──────────────────────────── Slot Management ────────────────────────────

    def _add_slot(self):
        idx = len(self.slots) + 1
        slot = ImageSlot(idx)
        slot.image_changed.connect(self._check_ready)
        slot.remove_requested.connect(self._remove_slot)
        self.slots.append(slot)
        self.slot_layout.insertWidget(self.slot_layout.count() - 1, slot)
        self._check_ready()

    def _remove_slot(self, slot: ImageSlot):
        if self.animating or self.export_worker is not None: return
        self.slot_layout.removeWidget(slot)
        self.slots.remove(slot)
        slot.deleteLater()
        for i, s in enumerate(self.slots):
            s.setTitle(f"Image {i+1}")
        self._check_ready()

    def _clear_all(self):
        if self.animating or self.export_worker is not None: return
        for slot in self.slots:
            self.slot_layout.removeWidget(slot)
            slot.deleteLater()
        self.slots.clear()
        self._add_slot()
        self._add_slot()
        self._check_ready()

    # ──────────────────────────── Logic ────────────────────────────

    def _check_ready(self):
        loaded_count = sum(1 for s in self.slots if s.image is not None)
        ready = loaded_count >= 2 and not self.animating and self.export_worker is None
        self.btn_start.setEnabled(ready)
        self.btn_export.setEnabled(loaded_count >= 2 and self.export_worker is None)

    def _get_target_size(self) -> tuple[int, int] | None:
        text = self.res_combo.currentText()
        if text == "Original": return None
        size = int(text)
        return size, size

    def _prepare_image_arrays(self) -> list[np.ndarray] | None:
        loaded_images = [s.image for s in self.slots if s.image is not None]
        if len(loaded_images) < 2:
            QMessageBox.warning(self, "Error", "Please load at least 2 images.")
            return None

        target = self._get_target_size()
        if target is None:
            w = min(img.width for img in loaded_images)
            h = min(img.height for img in loaded_images)
            target = (w, h)

        arrays = []
        for img in loaded_images:
            resized = img.convert("RGB").resize(target, Image.Resampling.LANCZOS)
            arrays.append(np.array(resized, dtype=np.uint8))
        return arrays

    def _start_morph(self):
        if self.animating: return

        arrays = self._prepare_image_arrays()
        if arrays is None: return

        self.image_arrays = arrays
        h, w, _ = self.image_arrays[0].shape
        self.total_pixels = h * w
        self.current_transition_idx = 0
        self._setup_transition()

        self.animating = True
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("⏸  Pause")
        self._update_timer_interval()
        self.timer.start()

    def _setup_transition(self):
        self.img_a_arr = self.image_arrays[self.current_transition_idx]
        self.img_b_arr = self.image_arrays[self.current_transition_idx + 1]
        self.current_arr = self.img_a_arr.copy()

        indices = np.arange(self.total_pixels, dtype=np.int32)
        np.random.shuffle(indices)
        self.shuffled_indices = indices
        self.cursor = 0

    def _update_timer_interval(self):
        val = self.speed_slider.value()
        interval = max(1, int(200 * (0.96 ** val)))
        self.timer.setInterval(interval)

    def _toggle_pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_pause.setText("▶  Resume")
        else:
            self._update_timer_interval()
            self.timer.start()
            self.btn_pause.setText("⏸  Pause")

    def _reset(self):
        self.timer.stop()
        self.animating = False
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸  Pause")
        self.current_arr = None
        self.image_arrays = []
        self.progress.setValue(0)
        self.canvas.clear()
        self.canvas.setText("Add images and press Start")
        self._check_ready()

    def _tick(self):
        if self.current_arr is None or self.shuffled_indices is None:
            return

        self._update_timer_interval()
        h, w, _ = self.current_arr.shape

        speed = self.speed_slider.value()
        batch = max(1, int(self.total_pixels * speed / 1500))
        end = min(self.cursor + batch, self.total_pixels)

        flat_idx = self.shuffled_indices[self.cursor:end]
        rows = flat_idx // w
        cols = flat_idx % w
        self.current_arr[rows, cols] = self.img_b_arr[rows, cols]

        self.cursor = end

        total_transitions = len(self.image_arrays) - 1
        total_pixels_all = self.total_pixels * total_transitions
        current_progress = (self.current_transition_idx * self.total_pixels) + self.cursor
        pct = int(100 * current_progress / total_pixels_all)
        self.progress.setValue(pct)

        self._update_canvas()

        if self.cursor >= self.total_pixels:
            self.current_transition_idx += 1
            if self.current_transition_idx >= total_transitions:
                if self.loop_check.isChecked():
                    self.current_transition_idx = 0
                else:
                    self.timer.stop()
                    self.animating = False
                    self.btn_pause.setEnabled(False)
                    self.btn_start.setEnabled(True)
                    return
            self._setup_transition()

    def _update_canvas(self):
        if self.current_arr is None: return
        pm = numpy_to_qpixmap(self.current_arr)
        canvas_size = self.canvas.size() - QSize(20, 20)
        scaled = pm.scaled(
            canvas_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.canvas.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_canvas()

    # ──────────────────────────── Export ────────────────────────────

    def _export_mp4(self):
        if imageio is None:
            QMessageBox.critical(
                self, "Missing Dependency",
                "imageio is not installed.\n\nRun:\n  pip install 'imageio[ffmpeg]'"
            )
            return

        arrays = self._prepare_image_arrays()
        if arrays is None: return

        dlg = ExportDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        fps, seconds_per_transition = dlg.values()

        path, _ = QFileDialog.getSaveFileName(
            self, "Save MP4", "morph.mp4", "MP4 Video (*.mp4)"
        )
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"

        self.export_status.setText("Starting export…")
        self.btn_export.setEnabled(False)
        self.btn_cancel_export.setEnabled(True)
        self.progress.setValue(0)

        self.export_worker = ExportWorker(arrays, path, fps, seconds_per_transition)
        self.export_worker.progress.connect(self.progress.setValue)
        self.export_worker.status.connect(self.export_status.setText)
        self.export_worker.finished_ok.connect(self._on_export_ok)
        self.export_worker.finished_err.connect(self._on_export_err)
        self.export_worker.finished.connect(self._on_export_finished)
        self.export_worker.start()

    def _cancel_export(self):
        if self.export_worker is not None:
            self.export_worker.cancel()
            self.export_status.setText("Cancelling…")

    def _on_export_ok(self, path: str):
        self.export_status.setText(f"✔ Saved: {path}")
        QMessageBox.information(self, "Export Complete", f"Video saved to:\n{path}")

    def _on_export_err(self, msg: str):
        self.export_status.setText("✘ Export failed")
        QMessageBox.critical(self, "Export Error", msg)

    def _on_export_finished(self):
        self.export_worker = None
        self.btn_cancel_export.setEnabled(False)
        self._check_ready()


def main():
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        QMainWindow, QWidget { background: #2b2b2b; color: #ddd; }
        QGroupBox { border: 1px solid #555; border-radius: 6px; margin-top: 10px; padding-top: 16px; font-weight: bold; }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
        QPushButton { background: #3a3a3a; border: 1px solid #666; border-radius: 4px; padding: 6px 14px; color: #eee; }
        QPushButton:hover { background: #505050; }
        QPushButton:disabled { color: #666; background: #2a2a2a; }
        QLineEdit { background: #1e1e1e; border: 1px solid #555; border-radius: 4px; padding: 4px; color: #eee; }
        QProgressBar { border: 1px solid #555; border-radius: 4px; text-align: center; background: #1e1e1e; color: #eee; }
        QProgressBar::chunk { background: #4caf50; border-radius: 3px; }
        QComboBox, QSpinBox, QDoubleSpinBox { background: #3a3a3a; border: 1px solid #666; border-radius: 4px; padding: 4px; color: #eee; }
        QScrollArea { border: 1px solid #444; border-radius: 4px; background: #222; }
        QScrollBar:vertical { background: #222; width: 10px; margin: 0px; }
        QScrollBar::handle:vertical { background: #555; border-radius: 5px; min-height: 20px; }
        QSlider::groove:horizontal { height: 6px; background: #555; border-radius: 3px; }
        QSlider::handle:horizontal { width: 16px; height: 16px; margin: -5px 0; background: #4caf50; border-radius: 8px; }
        QCheckBox { spacing: 8px; color: #ddd; }
        QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #666; border-radius: 3px; background: #1e1e1e; }
        QCheckBox::indicator:checked { background: #4caf50; border: 1px solid #4caf50; }
    """)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()