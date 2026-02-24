import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import config
from inference import CurrencyDetector
import time

class CurrencyDetectionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("800x700")

        self.cap = None
        self.detector = CurrencyDetector()
        self.running = False

        # GUI Components
        self.create_widgets()

        # Update loop
        self.update_frame()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # Title
        title_label = tk.Label(self.window, text="Currency Detection System", font=("Arial", 24, "bold"))
        title_label.pack(pady=10)

        # Video Canvas
        self.canvas = tk.Canvas(self.window, width=640, height=480, bg="black")
        self.canvas.pack(pady=10)

        # Results Frame
        res_frame = tk.Frame(self.window)
        res_frame.pack(pady=10)

        self.label_var = tk.StringVar(value="Detection: ---")
        self.conf_var = tk.StringVar(value="Confidence: ---%")
        self.fps_var = tk.StringVar(value="FPS: ---")

        tk.Label(res_frame, textvariable=self.label_var, font=("Arial", 16)).grid(row=0, column=0, padx=20)
        tk.Label(res_frame, textvariable=self.conf_var, font=("Arial", 16)).grid(row=0, column=1, padx=20)
        tk.Label(res_frame, textvariable=self.fps_var, font=("Arial", 12), fg="gray").grid(row=1, column=0, columnspan=2)

        # Controls
        ctrl_frame = tk.Frame(self.window)
        ctrl_frame.pack(pady=20)

        self.start_btn = ttk.Button(ctrl_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.grid(row=0, column=0, padx=10)

        self.stop_btn = ttk.Button(ctrl_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=10)

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(config.CAMERA_ID)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)

    def stop_camera(self):
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.canvas.delete("all")
            self.label_var.set("Detection: ---")
            self.conf_var.set("Confidence: ---%")
            self.fps_var.set("FPS: ---")

    def update_frame(self):
        if self.running and self.cap:
            start_time = time.time()
            ret, frame = self.cap.read()
            if ret:
                # Prediction
                label, confidence = self.detector.predict(frame)
                
                # Update UI
                self.label_var.set(f"Detection: {label}")
                self.conf_var.set(f"Confidence: {confidence*100:.2f}%")
                
                # Draw on frame
                h, w, _ = frame.shape
                start_p = (w // 4, h // 4)
                end_p = (3 * w // 4, 3 * h // 4)
                color = (0, 255, 0) if confidence > config.CONFIDENCE_THRESHOLD else (0, 0, 255)
                cv2.rectangle(frame, start_p, end_p, color, 2)
                
                # Convert to PIL/Tkinter
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((640, 480))
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas.image = imgtk
                
                # FPS
                fps = 1.0 / (time.time() - start_time)
                self.fps_var.set(f"FPS: {int(fps)}")

        self.window.after(10, self.update_frame)

    def on_closing(self):
        self.stop_camera()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CurrencyDetectionApp(root, "Currency Detection System")
    root.mainloop()
