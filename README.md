# Real-Time Face Recognition Dashboard

A web-based **Face Recognition System** built with Python, Streamlit, OpenCV, and the `face_recognition` library.

This application provides a modern dashboard for registering facial data, storing face encodings in a local SQLite database, and performing real-time face detection and recognition through a webcam.

The system is designed to keep the camera stream responsive while face recognition runs asynchronously in the background.

---

## Features

### Dashboard

* Total registered face data
* Total registered people
* Database status
* Face recognition engine information
* Monthly registration statistics
* Recently registered faces
* Registered face data table

### Face Registration

* Register a person's name and age
* Upload an existing image
* Capture an image directly from the webcam
* Automatically detect faces from the image
* Validate that exactly one face is present
* Generate face encoding
* Store facial data in SQLite
* Store registered images locally

### Real-Time Face Recognition

* Live webcam stream
* Real-time face detection
* Face recognition against registered data
* Bounding box around detected faces
* Display recognized person's name
* Display person's age
* Unknown face detection
* Camera remains active even when the database is empty

### Performance Optimization

The real-time recognition pipeline uses a background worker so that the video stream does not have to wait for the face recognition process.

The system uses:

* Background recognition thread
* Latest-frame processing
* Frame queue prevention
* Reduced-resolution AI processing
* HOG face detection model
* OpenCV frame processing
* Asynchronous Streamlit WebRTC processing

This architecture helps reduce video stuttering and recognition latency.

---

## Tech Stack

| Technology       | Purpose                                  |
| ---------------- | ---------------------------------------- |
| Python           | Main programming language                |
| Streamlit        | Web application framework                |
| OpenCV           | Image and video processing               |
| face_recognition | Face detection and face encoding         |
| dlib             | Face recognition backend                 |
| NumPy            | Numerical and image array processing     |
| Pandas           | Data processing and dashboard statistics |
| SQLite           | Local database                           |
| streamlit-webrtc | Real-time webcam streaming               |
| Pillow           | Image processing                         |

---

## System Architecture

```text
                    ┌──────────────────────┐
                    │       Streamlit      │
                    │      Web Dashboard   │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        ┌───────────┐    ┌───────────┐   ┌─────────────┐
        │ Dashboard │    │ Registration│   │ Recognition │
        └─────┬─────┘    └──────┬────┘   └──────┬──────┘
              │                 │               │
              │                 ▼               ▼
              │          Face Encoding      Webcam Stream
              │                 │               │
              │                 ▼               ▼
              └──────────► SQLite ◄──── Background Worker
                              │
                              ▼
                       Face Encodings
```

---

## Real-Time Recognition Flow

The recognition system separates the video pipeline from the AI recognition process.

```text
Webcam Frame
     │
     ▼
OpenCV Frame
     │
     ├──────────────► Display immediately
     │
     ▼
Resize for AI
     │
     ▼
Background Worker
     │
     ▼
Face Detection
     │
     ▼
Face Encoding
     │
     ▼
Compare with Database
     │
     ▼
Recognition Result
     │
     ▼
Bounding Box + Name + Age
```

The worker processes the **latest available frame** rather than building a large queue of old frames. This prevents the recognition process from falling behind the live camera stream.

---

## Project Structure

```text
real-time-face-recognition/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── data/
│   └── faces/
│       └── registered-face-images
│
├── face_database.db
│
└── .streamlit/
    └── config.toml
```

> `face_database.db` and registered face images contain personal/biometric data and should not be committed to a public repository.

Recommended `.gitignore`:

```gitignore
venv/
__pycache__/
*.pyc

face_database.db
data/faces/

.env
.streamlit/secrets.toml
```

---

## Requirements

Before running the application, make sure you have:

* Python 3.10+
* pip
* Webcam
* Git
* C++ Build Tools on Windows if required by `dlib`

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/fadlyfebrosp/real-time-face-recognition.git
```

Move into the project directory:

```bash
cd real-time-face-recognition
```

---

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `dlib` installation fails on Windows, install the required Microsoft C++ build tools and Windows SDK, then run the installation again.

---

## Running the Application

Run Streamlit with:

```bash
python -m streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## Using the Application

### 1. Dashboard

The Dashboard provides an overview of the face database and registration activity.

It displays:

* Total face records
* Total registered people
* Database status
* Recognition engine
* Registration statistics
* Recently registered faces

---

### 2. Register a Face

Navigate to:

```text
Registrasi Wajah
```

Enter:

```text
Nama Lengkap
Umur
```

Then select one of the available methods:

```text
Upload Foto
```

or

```text
Kamera
```

The system validates the image before saving it.

A valid registration must contain exactly **one face**.

The application then:

1. Detects the face.
2. Generates the face encoding.
3. Saves the image.
4. Stores the encoding in SQLite.
5. Adds the person's identity to the recognition database.

---

### 3. Real-Time Recognition

Navigate to:

```text
Real-time Recognition
```

Allow browser access to the webcam.

The system will continuously:

1. Capture webcam frames.
2. Detect faces.
3. Generate face encodings.
4. Compare them against registered face encodings.
5. Display recognition results.

Possible statuses:

```text
Tidak ada wajah terdeteksi
```

```text
Wajah terdeteksi • Tidak dikenali
```

```text
Wajah dikenali • Nama
```

When a face is recognized, the application displays:

```text
Nama | Umur tahun
```

along with a bounding box around the face.

---

## Face Recognition Method

The application uses the `face_recognition` library for face detection and facial encoding.

The real-time detector uses the **HOG (Histogram of Oriented Gradients)** model.

The general recognition process is:

```text
Image
  ↓
Face Detection
  ↓
Face Location
  ↓
Face Encoding
  ↓
Face Distance
  ↓
Tolerance Check
  ↓
Identity
```

The system compares the generated facial encoding against encodings stored in the SQLite database.

The current recognition tolerance is:

```python
tolerance = 0.48
```

A lower tolerance generally makes matching more strict and can reduce false positives, while potentially increasing false negatives.

---

## Database

The application uses SQLite for local data storage.

### `faces` table

| Column       | Type    | Description            |
| ------------ | ------- | ---------------------- |
| `id`         | INTEGER | Primary key            |
| `name`       | TEXT    | Person's name          |
| `age`        | INTEGER | Person's age           |
| `image_path` | TEXT    | Stored image location  |
| `encoding`   | BLOB    | Facial encoding        |
| `created_at` | TEXT    | Registration timestamp |

Face encodings are converted into binary data before being stored in SQLite.

---

## Performance Architecture

A normal real-time recognition pipeline can become slow when face recognition is executed directly inside the video frame callback.

This project avoids that design.

Instead:

```text
Video Thread
     │
     ├── Return frame immediately
     │
     └── Submit latest frame
                │
                ▼
       Background Worker
                │
                ├── Face Detection
                ├── Face Encoding
                └── Face Matching
                │
                ▼
        Recognition Result
                │
                ▼
        Overlay on Video
```

The worker always prioritizes the newest frame.

This prevents old frames from accumulating and causing delayed recognition results.

---

## Privacy and Security

This project processes facial data locally.

Registered facial data may include:

* Face images
* Facial encodings
* Names
* Ages

Because facial encodings are biometric-related data, avoid committing the following files to a public GitHub repository:

```text
face_database.db
data/faces/
```

For production use, additional security controls should be implemented, including:

* Authentication
* Authorization
* Encrypted storage
* Secure database access
* Access logging
* Data retention policy
* User consent
* Secure deployment configuration

---

## Troubleshooting

### Camera does not appear

Make sure:

1. The browser has permission to access the webcam.
2. No other application is currently using the webcam.
3. The correct camera is selected.
4. The application is running through Streamlit.

---

### `dlib` installation error

On Windows, `dlib` may require C++ build tools.

Install:

```text
Desktop development with C++
Windows SDK
```

Then reinstall the dependencies.

---

### Streamlit does not start

Try:

```bash
python -m streamlit run app.py
```

instead of:

```bash
streamlit run app.py
```

---

### Recognition is slow

Possible causes include:

* Low CPU performance
* Poor lighting
* Face positioned too far from the camera
* Multiple faces in the frame
* Large database
* High camera resolution

The application already separates video rendering from recognition using a background worker to reduce video stuttering.

---

## Future Improvements

Potential improvements for future versions:

* [ ] User authentication
* [ ] Recognition history
* [ ] Attendance tracking
* [ ] Recognition timestamp logging
* [ ] Multiple camera support
* [ ] Face recognition confidence score
* [ ] Face database management
* [ ] Delete/update registered faces from dashboard
* [ ] PostgreSQL/MySQL support
* [ ] REST API
* [ ] Docker deployment
* [ ] Cloud deployment
* [ ] GPU acceleration
* [ ] More advanced face detection models
* [ ] Recognition analytics
* [ ] Export recognition history to CSV/Excel

---

## Screenshots

Add application screenshots to the repository:

```text
screenshots/
├── dashboard.png
├── registration.png
└── recognition.png
```

Then display them in this README:

```markdown
## Screenshots

### Dashboard

![Dashboard](screenshots/dashboard.png)

### Face Registration

![Face Registration](screenshots/registration.png)

### Real-Time Recognition

![Real-Time Recognition](screenshots/recognition.png)
```

---

## Author

**Fadly Febro**

GitHub:
https://github.com/fadlyfebrosp

Repository:
https://github.com/fadlyfebrosp/real-time-face-recognition

---

## License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for more information.

---

## Disclaimer

This project is intended for educational, research, and portfolio purposes.

Facial recognition involves sensitive biometric information. Any real-world deployment should comply with applicable privacy, data protection, and biometric regulations.
