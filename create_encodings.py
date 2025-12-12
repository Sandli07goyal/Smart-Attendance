# create_encodings.py
import os
import pickle
import face_recognition

DATASET_DIR = "dataset"
ENCODINGS_PATH = "encodings.pkl"

def create_encodings(dataset_dir=DATASET_DIR, out_path=ENCODINGS_PATH):
    known_encodings = []
    known_names = []

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found. Create it and add images.")

    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"[INFO] Processing images for: {person_name}")
        count = 0
        for filename in os.listdir(person_dir):
            file_path = os.path.join(person_dir, filename)
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            try:
                image = face_recognition.load_image_file(file_path)
                encs = face_recognition.face_encodings(image)
                if len(encs) == 0:
                    print(f"  [WARN] No face found in {file_path}. Skipping.")
                    continue
                known_encodings.append(encs[0])
                known_names.append(person_name)
                count += 1
            except Exception as e:
                print(f"  [ERROR] Couldn't process {file_path}: {e}")

        print(f"  -> {count} valid images encoded for {person_name}.\n")

    if len(known_encodings) == 0:
        raise ValueError("No encodings found. Add images to the dataset and retry.")

    data = {"encodings": known_encodings, "names": known_names}
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[INFO] Encodings saved to: {out_path}")

if __name__ == "__main__":
    create_encodings()
