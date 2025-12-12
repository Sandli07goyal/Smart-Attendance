# 🧠 Smart Attendance System (Face Recognition)

A Machine Learning powered attendance automation system that uses OpenCV and face_recognition to detect and recognize faces in real-time using a webcam.
The system automatically marks attendance in an Excel sheet and saves captured face snapshots for verification.

# 🚀 Key Features

🔍 Real-time face detection using Haar Cascade
🧬 Accurate face recognition using dlib’s 128-D facial embeddings
📝 Automatic attendance logging (Name, Date, Time)
🖼 Face snapshot saving for proof-of-attendance
🔄 No duplicate attendance within the same session
📁 Clean and modular project structure
💻 Cross-platform support (Windows, macOS, Linux)

# 📁 Project Structure

Smart-attendance/
│
├── attendance_capture.py      # Main real-time attendance script
├── create_encodings.py        # Generate face encodings from dataset images
├── capture_images.py          # Capture face images using webcam
├── requirements.txt           # Project dependencies
│
├── dataset/                   # Training images (not included in repo)
│   └── PersonName/
│        ├── img1.jpg
│        ├── img2.jpg
│        └── ...
│
├── attendance_images/         # Saved snapshots of recognized faces
├── attendance.xlsx            # Generated attendance file
├── encodings.pkl              # Stored face encodings
│
└── venv/                      # Virtual environment (ignored in Git)

# ⚙️ How It Works (Short Overview)

Add multiple images for each person inside dataset/PersonName/.
Run create_encodings.py to generate face encodings (encodings.pkl).
Start attendance_capture.py → webcam opens → faces are detected & recognized.
When a face matches, the system:
Marks attendance in attendance.xlsx
Saves the face snapshot in attendance_images/
Press q to end the session and save the file.

# 🔧 Installation & Setup

# 1️⃣ Clone the Repository
git clone https://github.com/YOUR_USERNAME/Smart-Attendance.git
cd Smart-Attendance

# 2️⃣ Create & Activate Virtual Environment
macOS / Linux:
python3 -m venv venv
source venv/bin/activate

Windows:
python -m venv venv
.\venv\Scripts\activate

# 3️⃣ Install Required Libraries
pip install -r requirements.txt

# 🖼 Add Training Images
Create a folder for each person:
dataset/Sandli/
dataset/Person2/
Add 3–8 clear, front-facing images per person.
You can also capture images directly using webcam:
python capture_images.py Sandli 8

# 🧬 Generate Face Encodings
python create_encodings.py
This creates encodings.pkl which is used for matching.

# 🎥 Run the Attendance System
python attendance_capture.py
Webcam window will open
Detected faces will be labeled
Attendance is auto-recorded
Press q to quit

# 📊 Output Files
attendance.xlsx
Contains:
Name	Date	Time
attendance_images/
Stores snapshots like:
Sandli_20251212_163015.jpg

# 🛠️ Troubleshooting

# 🔹 Face Not Detected
Use clear, front-facing images
Ensure good lighting
Crop images around the face

# 🔹 "No encodings found"
Dataset folder is empty or contains images where faces aren’t detected.

# 🔹 Webcam not opening
Close other apps using the camera (Zoom, Google Meet, Teams).

# 🔐 Privacy Notice
Do NOT upload personal face images or encoded face data publicly.
Ensure .gitignore contains:
dataset/
attendance_images/
attendance.xlsx
encodings.pkl
venv/

# 🌟 Future Improvements
Web dashboard for viewing attendance
Database integration (SQLite / MongoDB)
Anti-spoofing / liveness detection
Email notifications for attendance
Mobile app for scanning faces
