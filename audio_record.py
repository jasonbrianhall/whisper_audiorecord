import sys
import torch
from whisper import load_model
import sounddevice as sd
import numpy as np
from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox, QTextEdit, QMessageBox
from PySide2.QtCore import Qt, QThread, Signal, QTimer
import signal

class AudioRecorder(QThread):
    finished = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, device, samplerate=16000):
        super().__init__()
        self.device = device
        self.samplerate = samplerate
        self.recording = False
        self.audio_data = []

    def run(self):
        try:
            with sd.InputStream(device=self.device, channels=1, samplerate=self.samplerate, callback=self.audio_callback):
                while self.recording:
                    sd.sleep(100)
            
            if self.audio_data:
                audio_array = np.concatenate(self.audio_data, axis=0)
                self.finished.emit(audio_array)
            else:
                self.error.emit("No audio data recorded")
        except sd.PortAudioError as e:
            self.error.emit(str(e))

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_data.append(indata.copy())

    def stop(self):
        self.recording = False

class WhisperTranscriber(QThread):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, audio_data):
        super().__init__()
        self.audio_data = audio_data

    def run(self):
        try:
            # Load the Whisper model
            model = load_model("base")
            
            # Prepare audio data
            audio_tensor = torch.from_numpy(self.audio_data.flatten()).float()
            audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))
            
            # Perform transcription
            result = model.transcribe(audio_tensor)
            self.finished.emit(result["text"])
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Transcription App")
        self.layout = QVBoxLayout()

        self.device_combo = QComboBox()
        self.update_device_list()
        self.layout.addWidget(self.device_combo)

        self.samplerate_combo = QComboBox()
        self.update_samplerate_list()
        self.layout.addWidget(self.samplerate_combo)

        self.record_button = QPushButton("Record")
        self.record_button.clicked.connect(self.toggle_recording)
        self.layout.addWidget(self.record_button)

        self.transcription_text = QTextEdit()
        self.transcription_text.setReadOnly(True)
        self.layout.addWidget(self.transcription_text)

        self.setLayout(self.layout)

        self.recorder = None
        self.is_recording = False

    def update_device_list(self):
        devices = sd.query_devices()
        self.device_combo.clear()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                self.device_combo.addItem(f"{dev['name']}", i)

    def update_samplerate_list(self):
        samplerates = [8000, 16000, 22050, 44100, 48000]
        for rate in samplerates:
            self.samplerate_combo.addItem(f"{rate} Hz", rate)

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        device = self.device_combo.currentData()
        samplerate = self.samplerate_combo.currentData()
        self.recorder = AudioRecorder(device, samplerate)
        self.recorder.finished.connect(self.on_recording_finished)
        self.recorder.error.connect(self.on_error)
        self.recorder.recording = True
        self.recorder.start()
        self.is_recording = True
        self.record_button.setText("Stop")

    def stop_recording(self):
        if self.recorder:
            self.recorder.stop()
            self.is_recording = False
            self.record_button.setText("Record")

    def on_recording_finished(self, audio_data):
        self.transcriber = WhisperTranscriber(audio_data)
        self.transcriber.finished.connect(self.on_transcription_finished)
        self.transcriber.error.connect(self.on_error)
        self.transcriber.start()

    def on_transcription_finished(self, text):
        self.transcription_text.setPlainText(text)

    def on_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)

    def closeEvent(self, event):
        self.stop_recording()
        if self.recorder:
            self.recorder.wait()
        event.accept()

class Application(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.window = MainWindow()
        self.window.show()

    def signal_handler(self, signum, frame):
        print("SIGINT received. Closing the application...")
        self.quit()

if __name__ == "__main__":
    app = Application(sys.argv)
    
    # Handle SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, app.signal_handler)
    
    # Use a timer to allow Python to process signals
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)
    
    sys.exit(app.exec_())
