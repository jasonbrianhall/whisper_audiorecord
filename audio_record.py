import sys
import whisper
import sounddevice as sd
import numpy as np
import signal
import tempfile
#from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox, QTextEdit, QMessageBox
#from PySide2.QtCore import Qt, QThread, Signal, QTimer
from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox, QTextEdit, QMessageBox
from PySide2.QtCore import Qt, QThread, Signal as QtSignal, QTimer
import soundfile as sf

class AudioRecorder(QThread):
    finished = QtSignal(np.ndarray)
    error = QtSignal(str)

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
    finished = QtSignal(str)
    language_detected = QtSignal(str)
    error = QtSignal(str)

    def __init__(self, audio_data, sample_rate, model_name):
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.model_name = model_name
        self.model = None

    def run(self):
        try:
            # Load the selected model
            self.model = whisper.load_model(self.model_name)

            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                sf.write(temp_wav.name, self.audio_data, self.sample_rate)
                temp_wav_path = temp_wav.name

            # Transcribe the audio using the temporary file path
            result = self.model.transcribe(temp_wav_path)

            # Emit the detected language
            self.language_detected.emit(f"Detected language: {result['language']}")

            # Emit the transcribed text
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
        self.device_combo.currentIndexChanged.connect(self.update_samplerate_list)
        self.layout.addWidget(self.device_combo)

        self.samplerate_combo = QComboBox()
        self.update_samplerate_list()
        self.layout.addWidget(self.samplerate_combo)

        # Add model selection combo box
        self.model_combo = QComboBox()
        self.update_model_list()
        self.layout.addWidget(self.model_combo)

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
        device_id = self.device_combo.currentData()
        if device_id is not None:
            device_info = sd.query_devices(device_id, 'input')
            supported_samplerates = self.get_supported_samplerates(device_info)
            self.samplerate_combo.clear()
            for rate in supported_samplerates:
                self.samplerate_combo.addItem(f"{rate} Hz", rate)

    def get_supported_samplerates(self, device_info):
        supported = []
        for rate in [8000, 16000, 22050, 44100, 48000]:
            try:
                sd.check_input_settings(device=device_info['index'], samplerate=rate)
                supported.append(rate)
            except sd.PortAudioError:
                pass
        return supported

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        device = self.device_combo.currentData()
        samplerate = self.samplerate_combo.currentData()
        if device is None or samplerate is None:
            QMessageBox.warning(self, "Error", "Please select a device and sample rate.")
            return
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
        sample_rate = self.samplerate_combo.currentData()
        self.transcriber = WhisperTranscriber(audio_data, sample_rate)
        self.transcriber.finished.connect(self.on_transcription_finished)
        self.transcriber.error.connect(self.on_error)
        self.transcriber.start()

    def on_transcription_finished(self, text):
        self.transcription_text.setPlainText(text)
        print("Transcription:")
        print(text)  # Print the transcription to the CLI

    def on_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)

    def closeEvent(self, event):
        self.stop_recording()
        if self.recorder:
            self.recorder.wait()
        event.accept()
        
    def update_model_list(self):
        models = ["tiny", "base", "small", "medium", "large"]
        for model in models:
            self.model_combo.addItem(model)

    def on_recording_finished(self, audio_data):
        sample_rate = self.samplerate_combo.currentData()
        model_name = self.model_combo.currentText()
        self.transcriber = WhisperTranscriber(audio_data, sample_rate, model_name)
        self.transcriber.finished.connect(self.on_transcription_finished)
        self.transcriber.error.connect(self.on_error)
        self.transcriber.start()

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
