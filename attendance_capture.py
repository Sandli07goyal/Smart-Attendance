# attendance_capture.py
import os
import pickle
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
import face_recognition
import imutils

# ---------------- CONFIG ----------------
ENCODINGS_PATH = "encodings.pkl"
ATTENDANCE_FILE = "attendance.xlsx"
ATTENDANCE_IMG_DIR = "attendance_images"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# recognition params
TOLERANCE = 0.5                # lower -> stricter matching (0.4-0.6 typical)
CONFIRMATION_FRAMES = 3        # frames required in a row to mark attendance
RESIZE_WIDTH = 640             # resize frame width for speed
SNAPSHOT_AFTER_MARK = True     # save a face snapshot when marked

# ----------------------------------------

def load_known(enc_path=ENCODINGS_PATH):
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"Encodings file '{enc_path}' not found. Run create_encodings.py first.")
    with open(enc_path, "rb") as f:
        data = pickle.load(f)
    known_encodings = data.get("encodings", [])
    known_names = data.get("names", [])
    if len(known_encodings) == 0:
        raise ValueError("No known encodings found in the encodings file.")
    print(f"[INFO] Loaded {len(known_encodings)} known face encodings.")
    return known_encodings, known_names

def ensure_dirs():
    if not os.path.exists(ATTENDANCE_IMG_DIR):
        os.makedirs(ATTENDANCE_IMG_DIR, exist_ok=True)

def main():
    known_encodings, known_names = load_known()
    ensure_dirs()
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    attendance_rows = []
    marked_names = set()
    confirmation_counts = {}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Exiting.")
        return

    print("[INFO] Starting video stream. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame = imutils.resize(frame, width=RESIZE_WIDTH)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        detected_names_this_frame = []

        for (x, y, w, h) in faces:
            pad_w = int(0.15 * w)
            pad_h = int(0.15 * h)
            x1 = max(x - pad_w, 0)
            y1 = max(y - pad_h, 0)
            x2 = min(x + w + pad_w, frame.shape[1])
            y2 = min(y + h + pad_h, frame.shape[0])

            face_rgb = rgb_frame[y1:y2, x1:x2]
            try:
                encs = face_recognition.face_encodings(face_rgb)
            except Exception:
                encs = []

            name = "Unknown"
            if encs:
                enc = encs[0]
                distances = face_recognition.face_distance(known_encodings, enc)
                best_idx = np.argmin(distances)
                best_distance = distances[best_idx]
                if best_distance <= TOLERANCE:
                    name = known_names[best_idx]
                else:
                    name = "Unknown"

            detected_names_this_frame.append((name, (x1, y1, x2, y2), face_rgb))
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        names_detected = [n for (n, bbox, face) in detected_names_this_frame if n != "Unknown"]
        for n in names_detected:
            confirmation_counts[n] = confirmation_counts.get(n, 0) + 1
        for name in list(confirmation_counts.keys()):
            if name not in names_detected:
                confirmation_counts[name] = 0

        for (name, bbox, face_rgb) in detected_names_this_frame:
            if name == "Unknown":
                continue
            if name in marked_names:
                continue
            if confirmation_counts.get(name, 0) >= CONFIRMATION_FRAMES:
                ts = datetime.now()
                row = {"Name": name, "Date": ts.date().isoformat(), "Time": ts.time().strftime("%H:%M:%S")}
                attendance_rows.append(row)
                marked_names.add(name)
                print(f"[MARKED] {name} at {row['Date']} {row['Time']}")

                if SNAPSHOT_AFTER_MARK:
                    x1, y1, x2, y2 = bbox
                    face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
                    filename = f"{name}_{ts.strftime('%Y%m%d_%H%M%S')}.jpg"
                    filepath = os.path.join(ATTENDANCE_IMG_DIR, filename)
                    cv2.imwrite(filepath, face_bgr)
                    print(f"  -> Snapshot saved to {filepath}")

        info_text = f"Marked: {len(marked_names)}"
        cv2.putText(frame, info_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Smart Attendance", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(attendance_rows) > 0:
        df = pd.DataFrame(attendance_rows)
        if os.path.exists(ATTENDANCE_FILE):
            try:
                existing = pd.read_excel(ATTENDANCE_FILE)
                df = pd.concat([existing, df], ignore_index=True)
            except Exception:
                pass
        df.to_excel(ATTENDANCE_FILE, index=False)
        print(f"[INFO] Attendance saved to {ATTENDANCE_FILE}")
    else:
        print("[INFO] No attendance to save for this session.")

if __name__ == "__main__":
    main()
