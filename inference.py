import cv2
import numpy as np
import tensorflow as tf
import config
import time

class CurrencyDetector:
    def __init__(self, model_path=config.MODEL_PATH):
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            
        self.classes = config.CLASSES

    def preprocess_frame(self, frame):
        # Resize to model input size
        img = cv2.resize(frame, config.IMG_SIZE)
        # Normalize
        img = img.astype('float32') / 255.0
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, frame):
        if self.model is None:
            return "No Model", 0.0
            
        processed_img = self.preprocess_frame(frame)
        predictions = self.model.predict(processed_img, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        return self.classes[class_idx], confidence

def run_realtime():
    detector = CurrencyDetector()
    cap = cv2.VideoCapture(config.CAMERA_ID)
    
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Prediction
        label, confidence = detector.predict(frame)
        
        # Display Results
        display_text = f"{label}: {confidence*100:.2f}% (FPS: {int(fps)})"
        
        # Draw bounding box (simulated for classification model)
        # For a simple classification model, we draw a box around the center
        h, w, _ = frame.shape
        start_point = (w // 4, h // 4)
        end_point = (3 * w // 4, 3 * h // 4)
        
        color = (0, 255, 0) if confidence > config.CONFIDENCE_THRESHOLD else (0, 0, 255)
        cv2.rectangle(frame, start_point, end_point, color, 2)
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow("Currency Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime()
