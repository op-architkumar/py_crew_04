import cv2
import numpy as np
from datetime import datetime

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def fake_thermal_effect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return colored, gray

def map_gray_to_celsius(gray_val, min_temp=30.0, max_temp=40.0):
    return min_temp + (gray_val / 255.0) * (max_temp - min_temp)

def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def log_temperature(celsius, fahrenheit):
    with open("temperature_log.csv", "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp},{celsius:.2f},{fahrenheit:.2f}\n")

def detect_forehead_temperature(frame, gray_frame, faces, log=True, show_fahrenheit=True):
    output = frame.copy()
    for (x, y, w, h) in faces:
        # Estimate forehead region
        forehead_y_start = y
        forehead_y_end = y + int(h * 0.2)
        forehead_x_start = x + int(w * 0.3)
        forehead_x_end = x + int(w * 0.7)

        forehead_roi = gray_frame[forehead_y_start:forehead_y_end, forehead_x_start:forehead_x_end]
        if forehead_roi.size == 0:
            continue

        avg_gray_val = np.mean(forehead_roi)
        temp_celsius = map_gray_to_celsius(avg_gray_val)
        temp_fahrenheit = celsius_to_fahrenheit(temp_celsius)

        # Log temp to file
        if log:
            log_temperature(temp_celsius, temp_fahrenheit)

        # Fever warning
        color = (0, 255, 0)  # Green
        if temp_celsius > 37.5:
            color = (0, 0, 255)  # Red

        # Draw forehead box
        cv2.rectangle(output, (forehead_x_start, forehead_y_start),
                      (forehead_x_end, forehead_y_end), color, 2)

        # Display temperature
        temp_text = f"{temp_celsius:.2f}°C"
        if show_fahrenheit:
            temp_text += f" / {temp_fahrenheit:.2f}°F"

        cv2.putText(output, temp_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if temp_celsius > 37.5:
            cv2.putText(output, "FEVER DETECTED!", (x, y + h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return output

def start_live_forehead_temp_detection():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the front camera.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        thermal_frame, gray_frame = fake_thermal_effect(frame)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        result_frame = detect_forehead_temperature(thermal_frame, gray_frame, faces)

        cv2.imshow("Forehead Temperature Detector", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_live_forehead_temp_detection()