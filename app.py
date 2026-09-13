import os
import time
import uuid
import sqlite3
import threading
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import face_recognition

from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "Face Recognition Dashboard"

DATABASE_PATH = "face_database.db"
FACE_DIRECTORY = "data/faces"

os.makedirs(FACE_DIRECTORY, exist_ok=True)


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    """
    Membuat koneksi ke SQLite database.
    """
    return sqlite3.connect(DATABASE_PATH, check_same_thread=False)


def initialize_database():
    """
    Membuat tabel faces jika belum tersedia.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            encoding BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


# ============================================================
# FACE DATABASE FUNCTIONS
# ============================================================

def save_face(name, age, image, encoding):
    """
    Menyimpan:
    - nama
    - umur
    - foto
    - face encoding

    ke database dan storage lokal.
    """

    unique_id = uuid.uuid4().hex

    image_filename = f"{unique_id}.jpg"
    image_path = os.path.join(
        FACE_DIRECTORY,
        image_filename
    )

    # Simpan image
    image.save(image_path)

    # Convert encoding menjadi bytes agar bisa disimpan SQLite
    encoding_blob = encoding.astype(np.float64).tobytes()

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO faces
        (name, age, image_path, encoding, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            name,
            age,
            image_path,
            encoding_blob,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    connection.commit()
    connection.close()


def load_faces():
    """
    Mengambil seluruh data wajah dari database.
    """

    connection = get_connection()

    dataframe = pd.read_sql_query(
        """
        SELECT
            id,
            name,
            age,
            image_path,
            encoding,
            created_at
        FROM faces
        ORDER BY id DESC
        """,
        connection
    )

    connection.close()

    return dataframe


def delete_face(face_id, image_path):
    """
    Menghapus data wajah dari:
    - SQLite
    - folder gambar
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM faces WHERE id = ?",
        (face_id,)
    )

    connection.commit()
    connection.close()

    # Hapus file gambar
    if os.path.exists(image_path):
        os.remove(image_path)


# ============================================================
# FACE ENCODING
# ============================================================

def generate_face_encoding(image):
    """
    Menghasilkan face encoding dari sebuah gambar.

    Return:
        encoding jika tepat satu wajah ditemukan
        None jika tidak ditemukan / lebih dari satu wajah
    """

    # Convert PIL Image -> NumPy
    image_array = np.array(image)

    # RGB -> RGB
    rgb_image = image_array

    # Deteksi wajah
    face_locations = face_recognition.face_locations(
        rgb_image,
        model="hog"
    )

    if len(face_locations) == 0:
        return None, "Tidak ada wajah yang ditemukan."

    if len(face_locations) > 1:
        return None, "Terdapat lebih dari satu wajah. Gunakan foto dengan satu wajah."

    # Generate encoding
    encodings = face_recognition.face_encodings(
        rgb_image,
        face_locations
    )

    if not encodings:
        return None, "Face encoding gagal dibuat."

    return encodings[0], None


# ============================================================
# LOAD KNOWN FACES
# ============================================================

def load_known_faces():
    """
    Mengambil seluruh face encoding dari database.
    """

    dataframe = load_faces()

    known_encodings = []
    known_names = []
    known_ages = []

    for _, row in dataframe.iterrows():

        encoding = np.frombuffer(
            row["encoding"],
            dtype=np.float64
        )

        known_encodings.append(encoding)
        known_names.append(row["name"])
        known_ages.append(row["age"])

    return (
        known_encodings,
        known_names,
        known_ages
    )


# ============================================================
# REAL-TIME FACE RECOGNITION
# ============================================================

class FaceRecognitionProcessor(VideoProcessorBase):

    def __init__(self):
        # ====================================================
        # DATABASE
        # ====================================================
        self.known_encodings = []
        self.known_names = []
        self.known_ages = []
        self.reload_database()

        # ====================================================
        # DISPLAY / RECOGNITION STATE
        # ====================================================
        self.status = "Kamera aktif • Menunggu wajah..."
        self.last_match = None
        self.last_faces = []

        # ====================================================
        # PERFORMANCE
        # ====================================================
        # Kamera tetap menggunakan resolusi tinggi untuk preview.
        # Recognition dilakukan pada frame yang lebih kecil.
        self.processing_scale = 0.40

        # Worker tidak dijalankan berdasarkan nomor frame.
        # Setiap kali worker selesai, frame TERBARU akan diproses.
        self.latest_frame = None
        self.worker_busy = False
        self.worker_lock = threading.Lock()
        self.state_lock = threading.Lock()

        # Thread recognition dibuat satu kali dan hidup selama kamera aktif.
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )
        self.worker_thread.start()

    def reload_database(self):
        (
            self.known_encodings,
            self.known_names,
            self.known_ages
        ) = load_known_faces()

    def _worker_loop(self):
        """
        Background recognition loop.

        Penting:
        - Tidak pernah dipanggil langsung dari recv().
        - Selalu mengambil frame PALING BARU.
        - Frame lama dibuang agar latency tidak menumpuk.
        """
        while True:
            frame_to_process = None

            with self.worker_lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame
                    self.latest_frame = None
                    self.worker_busy = True

            if frame_to_process is None:
                time.sleep(0.005)
                continue

            try:
                self._recognize(frame_to_process)
            except Exception as exc:
                with self.state_lock:
                    self.status = f"Recognition error: {exc}"
                    self.last_faces = []
                    self.last_match = None
            finally:
                with self.worker_lock:
                    self.worker_busy = False

    def _submit_latest_frame(self, rgb_small):
        """
        Masukkan hanya frame terbaru ke worker.

        Jika worker sedang sibuk, frame lama tidak diantrikan.
        Ini mencegah video recognition menjadi tertinggal/delay.
        """
        with self.worker_lock:
            self.latest_frame = rgb_small

    def _recognize(self, rgb_small):
        face_locations = face_recognition.face_locations(
            rgb_small,
            number_of_times_to_upsample=0,
            model="hog"
        )

        face_encodings = face_recognition.face_encodings(
            rgb_small,
            face_locations
        )

        current_faces = []
        new_status = "Tidak ada wajah terdeteksi"
        new_match = None

        if not face_locations:
            with self.state_lock:
                self.status = new_status
                self.last_match = None
                self.last_faces = []
            return

        scale_back = 1 / self.processing_scale

        for face_encoding, location in zip(face_encodings, face_locations):
            top, right, bottom, left = location

            top = int(top * scale_back)
            right = int(right * scale_back)
            bottom = int(bottom * scale_back)
            left = int(left * scale_back)

            name = "Unknown"
            age = "-"

            if self.known_encodings:
                distances = face_recognition.face_distance(
                    self.known_encodings,
                    face_encoding
                )

                best_match_index = int(np.argmin(distances))
                tolerance = 0.48

                if distances[best_match_index] <= tolerance:
                    name = self.known_names[best_match_index]
                    age = self.known_ages[best_match_index]
                    new_status = f"Wajah dikenali • {name} • {age} tahun"
                    new_match = name
                else:
                    new_status = "Wajah terdeteksi • Tidak dikenali"
            else:
                new_status = "Wajah terdeteksi • Tidak dikenali"

            current_faces.append({
                "location": (top, right, bottom, left),
                "name": name,
                "age": age
            })

        # Update state sekaligus supaya overlay selalu konsisten.
        with self.state_lock:
            self.status = new_status
            self.last_match = new_match
            self.last_faces = current_faces

    def recv(self, frame):
        """
        Jalur video dibuat seringan mungkin.

        recv():
        1. Ambil frame kamera.
        2. Kirim COPY frame terbaru ke background worker.
        3. Overlay hasil recognition terakhir.
        4. Langsung return frame.

        Tidak ada face_recognition() dan tidak ada sleep()
        di jalur video.
        """
        image = frame.to_ndarray(format="bgr24")

        # Kirim frame terbaru secara non-blocking.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        small_rgb = cv2.resize(
            rgb_image,
            None,
            fx=self.processing_scale,
            fy=self.processing_scale,
            interpolation=cv2.INTER_AREA
        )

        self._submit_latest_frame(small_rgb)

        # Ambil snapshot state secara singkat.
        with self.state_lock:
            faces_snapshot = list(self.last_faces)
            status_snapshot = self.status

        # ====================================================
        # DRAW STATUS
        # ====================================================
        if not faces_snapshot:
            if status_snapshot == "Tidak ada wajah terdeteksi":
                cv2.rectangle(
                    image,
                    (18, 18),
                    (330, 58),
                    (35, 42, 55),
                    cv2.FILLED
                )

                cv2.circle(
                    image,
                    (36, 38),
                    6,
                    (0, 180, 255),
                    cv2.FILLED
                )

                cv2.putText(
                    image,
                    "Tidak ada wajah terdeteksi",
                    (52, 44),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        # ====================================================
        # DRAW LAST RECOGNITION RESULT
        # ====================================================
        for face in faces_snapshot:
            top, right, bottom, left = face["location"]
            name = face["name"]
            age = face["age"]

            if name == "Unknown":
                status_text = "Wajah terdeteksi • Tidak dikenali"
                status_bg = (45, 100, 210)
            else:
                status_text = f"Wajah dikenali • {name}"
                status_bg = (0, 155, 105)

            text_width = max(
                250,
                min(650, 16 * len(status_text) + 35)
            )

            cv2.rectangle(
                image,
                (18, 18),
                (18 + text_width, 58),
                status_bg,
                cv2.FILLED
            )

            cv2.putText(
                image,
                status_text,
                (31, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Bounding box.
            cv2.rectangle(
                image,
                (left, top),
                (right, bottom),
                (0, 200, 120),
                2
            )

            # Label nama dan umur.
            label = f"{name} | {age} tahun"
            label_height = 35

            cv2.rectangle(
                image,
                (left, bottom),
                (right, bottom + label_height),
                (0, 200, 120),
                cv2.FILLED
            )

            cv2.putText(
                image,
                label,
                (left + 7, bottom + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Face Recognition | Dashboard",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# INITIALIZATION
# ============================================================

initialize_database()


# ============================================================
# SIDEBAR STATE
# ============================================================

if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

def toggle_sidebar():
    st.session_state.sidebar_open = not st.session_state.sidebar_open


# ============================================================
# MODERN UI
# ============================================================

st.markdown(
            """
    <style>
    /* ---------- Global ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    [data-testid="stAppViewContainer"] {
        background: #f5f7fb;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1380px;
        padding: 1.7rem 2.4rem 1rem 2.4rem;
    }

    /* ---------- Sidebar ---------- */

    /* Smooth hide/show animation */
    [data-testid="stSidebar"] {
        transition: all 0.25s ease;
    }

    /* Sidebar hidden state */
    body.sidebar-hidden [data-testid="stSidebar"] {
        display: none !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102a56 0%, #0b2146 100%);
        border-right: none;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1.1rem;
    }

    [data-testid="stSidebar"] * {
        color: #eef5ff;
    }

    .brand {
        padding: 0.3rem 0.4rem 1.4rem 0.4rem;
    }

    .brand-icon {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255,255,255,0.13);
        font-size: 23px;
        margin-bottom: 13px;
    }

    .brand-title {
        font-size: 19px;
        font-weight: 750;
        letter-spacing: -0.3px;
    }

    .brand-subtitle {
        margin-top: 4px;
        font-size: 12px;
        color: #aebed8;
    }

    .profile-card {
        margin: 0.5rem 0 1.3rem 0;
        padding: 14px;
        border-radius: 16px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.09);
    }

    .profile-row {
        display: flex;
        align-items: center;
        gap: 11px;
    }

    .profile-avatar {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: #FFFFFF;
        color: #102a56;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        font-weight: 800;
    }

    .profile-name {
        font-size: 13px;
        font-weight: 700;
    }

    .profile-role {
        font-size: 11px;
        color: #b7c6dc;
        margin-top: 2px;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 10px 12px;
        margin: 3px 0;
        border-radius: 11px;
        font-size: 13px;
        transition: 0.2s ease;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255,255,255,0.09);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: #FFFFFF;
        color: #102a56 !important;
        font-weight: 700;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) * {
        color: #102a56 !important;
    }

    /* ---------- Typography ---------- */
    .eyebrow {
        color: #6d7b91;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .page-title {
        color: #172033;
        font-size: 32px;
        line-height: 1.1;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0;
    }

    .page-subtitle {
        color: #758196;
        font-size: 14px;
        margin-top: 8px;
        margin-bottom: 1.6rem;
    }

    /* ---------- Header ---------- */
    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 7px 11px;
        border-radius: 999px;
        background: #e9f8ef;
        color: #218653;
        font-size: 12px;
        font-weight: 700;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #3BA77C;
    }

    /* ---------- Cards ---------- */
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E3E9F2;
        border-radius: 18px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 8px 25px rgba(25, 45, 75, 0.055);
    }

    .stat-card:hover {
        box-shadow: 0 12px 30px rgba(25, 45, 75, 0.09);
    }

    .stat-icon {
        width: 38px;
        height: 38px;
        border-radius: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #edf4ff;
        font-size: 18px;
        margin-bottom: 13px;
    }

    .stat-label {
        color: #7b879a;
        font-size: 12px;
        font-weight: 600;
    }

    .stat-value {
        color: #182236;
        font-size: 27px;
        line-height: 1;
        font-weight: 800;
        margin-top: 6px;
    }

    .stat-caption {
        color: #9aa5b5;
        font-size: 11px;
        margin-top: 8px;
    }

    /* ---------- Section Cards ---------- */
    .panel {
        background: #FFFFFF;
        border: 1px solid #E3E9F2;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 8px 25px rgba(25, 45, 75, 0.045);
    }

    .panel-title {
        color: #1a2437;
        font-size: 15px;
        font-weight: 750;
    }

    .panel-subtitle {
        color: #7B8798;
        font-size: 11px;
        margin-top: 3px;
        margin-bottom: 16px;
    }

    /* ---------- Recent List ---------- */
    .recent-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 11px 0;
        border-bottom: 1px solid #E9EEF5;
    }

    .recent-item:last-child {
        border-bottom: none;
    }

    .face-avatar {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        background: #edf4ff;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #2b67c7;
        font-weight: 800;
    }

    .recent-name {
        color: #273247;
        font-size: 12px;
        font-weight: 700;
    }

    .recent-meta {
        color: #929dad;
        font-size: 10px;
        margin-top: 3px;
    }

    /* ---------- Buttons / Inputs ---------- */
    .stButton > button {
        border-radius: 11px;
        min-height: 42px;
        font-weight: 700;
        border: 1px solid #dce4ef;
    }

    .stButton > button[kind="primary"] {
        background: #173b72;
        border-color: #173b72;
    }

    [data-testid="stFileUploader"] {
        border-radius: 14px;
    }

    /* ---------- Dataframe ---------- */
    [data-testid="stDataFrame"] {
        border: 1px solid #DDE5EF;
        border-radius: 14px;
        overflow: hidden;
        background: #FFFFFF;
    }

    [data-testid="stDataFrame"] iframe {
        border-radius: 14px;
    }

    /* Native chart spacing */
    [data-testid="stVegaLiteChart"] {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 8px 8px 0 0;
        border: 1px solid #E9EEF5;
    }

    /* ---------- Sidebar toggle button ---------- */
    [data-testid="stButton"] button {
        border-radius: 11px;
    }

    /* Double-chevron sidebar control */
    .stButton button[kind="secondary"] {
        font-size: 22px;
        font-weight: 700;
        line-height: 1;
    }

    /* ---------- Recognition Camera ---------- */
    [data-testid="stCustomComponentV1"] video {
        width: 100% !important;
        height: auto !important;
        max-height: 68vh !important;
        object-fit: contain !important;
        object-position: center center !important;
        background: #172B4D !important;
        border-radius: 14px;
    }

    /* streamlit-webrtc may render the video without the custom component
       wrapper depending on its version. */
    video {
        object-fit: contain !important;
        object-position: center center !important;
    }

    .camera-note {
        max-width: 100%;
        box-sizing: border-box;
    }

    /* ---------- Camera Status ---------- */
    .camera-status-guide {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        width: 100%;
        margin: 0 0 14px 0;
    }

    .guide-item {
        display: flex;
        align-items: center;
        gap: 9px;
        min-height: 58px;
        padding: 10px 12px;
        background: #FFFFFF;
        border: 1px solid #E3E9F2;
        border-radius: 13px;
        box-sizing: border-box;
    }

    .guide-dot {
        width: 8px;
        height: 8px;
        min-width: 8px;
        border-radius: 50%;
    }

    .guide-dot.waiting {
        background: #D7A83E;
    }

    .guide-dot.detected {
        background: #4F7CC7;
    }

    .guide-dot.recognized {
        background: #3BA77C;
    }

    .guide-item strong {
        display: block;
        color: #172B4D;
        font-size: 11px;
        line-height: 1.25;
    }

    .guide-item small {
        display: block;
        color: #7B8798;
        font-size: 9px;
        line-height: 1.3;
        margin-top: 2px;
    }

    .camera-note {
        margin-top: 12px;
        padding: 13px 15px;
        border-radius: 13px;
        background: #FFFFFF;
        border: 1px solid #E3E9F2;
        color: #64748B;
        font-size: 10px;
        line-height: 1.65;
        box-shadow: 0 5px 18px rgba(25, 45, 75, 0.035);
    }

    .camera-note strong {
        color: #334155;
    }

    /* ---------- Soft Info ---------- */
    .soft-info {
        margin-top: 10px;
        padding: 12px 14px;
        border: 1px solid #DCE5F0;
        border-radius: 12px;
        background: #F7F9FC;
        color: #64748B;
        font-size: 10px;
        line-height: 1.6;
    }

    .soft-info strong {
        color: #334155;
    }

    /* ---------- Footer ---------- */
    .footer {
        margin-top: 40px;
        padding: 18px 0 8px;
        border-top: 1px solid #e4e9f0;
        color: #98a2b1;
        text-align: center;
        font-size: 11px;
    }

    /* ---------- Responsive Layout ---------- */
    @media (max-width: 1100px) {
        .block-container {
            padding: 1.35rem 1.25rem 1rem 1.25rem;
        }

        .page-title {
            font-size: 29px;
        }

        .stat-card {
            padding: 16px;
        }

        .stat-value {
            font-size: 23px;
        }
    }

    @media (max-width: 900px) {
        [data-testid="stCustomComponentV1"] video,
        video {
            max-height: 55vh !important;
        }


        .block-container {
            padding: 1rem 0.85rem 1rem 0.85rem;
        }

        .page-title {
            font-size: 26px;
            letter-spacing: -0.6px;
        }

        .page-subtitle {
            font-size: 12px;
            margin-bottom: 1rem;
        }

        .stat-card {
            min-height: 105px;
            padding: 15px;
            border-radius: 15px;
        }

        .stat-icon {
            width: 34px;
            height: 34px;
            margin-bottom: 9px;
        }

        .stat-value {
            font-size: 21px;
        }

        .stat-caption {
            font-size: 10px;
        }

        .panel {
            padding: 16px;
            border-radius: 15px;
        }

        .status-pill {
            font-size: 10px;
            padding: 6px 9px;
        }

        .topbar {
            align-items: flex-start;
            gap: 10px;
        }

        [data-testid="stButton"] button {
            min-width: 42px;
            min-height: 42px;
            font-size: 24px;
            font-weight: 800;
        }
    }

    @media (max-width: 900px) {
        .camera-status-guide {
            grid-template-columns: 1fr;
        }

        .guide-item {
            min-height: 52px;
        }
    }

    @media (max-width: 640px) {
        .camera-status-guide {
            display: block;
        }

        .guide-item {
            margin-bottom: 7px;
        }

        .block-container {
            padding: 0.8rem 0.65rem 0.8rem 0.65rem;
        }

        .page-title {
            font-size: 23px;
        }

        .eyebrow {
            font-size: 10px;
        }

        .page-subtitle {
            font-size: 11px;
        }

        .stat-card {
            min-height: 96px;
            padding: 12px;
        }

        .stat-label {
            font-size: 9px;
        }

        .stat-value {
            font-size: 18px;
        }

        .stat-caption {
            display: none;
        }

        .panel-title {
            font-size: 13px;
        }

        .panel-subtitle {
            font-size: 10px;
        }

        .footer {
            font-size: 9px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

# Responsive sidebar:
# - Desktop: sidebar tetap memiliki lebar normal ketika terbuka.
# - Closed: sidebar dibuat selebar 0 agar state Streamlit tidak "hilang"
#   permanen dan tombol » tetap berada di area utama.
if st.session_state.sidebar_open:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            min-width: 285px !important;
            max-width: 285px !important;
            transform: translateX(0) !important;
            visibility: visible !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            width: 285px !important;
        }

        @media (max-width: 900px) {
            [data-testid="stSidebar"] {
                min-width: min(285px, 86vw) !important;
                max-width: min(285px, 86vw) !important;
            }

            [data-testid="stSidebar"] > div:first-child {
                width: min(285px, 86vw) !important;
            }
        }
        </style>
            """,
            unsafe_allow_html=True
        )
else:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            min-width: 0 !important;
            max-width: 0 !important;
            width: 0 !important;
            transform: translateX(-105%) !important;
            visibility: hidden !important;
            opacity: 0 !important;
            overflow: hidden !important;
            pointer-events: none !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            width: 0 !important;
            padding: 0 !important;
        }

        [data-testid="stSidebarCollapseButton"] {
            display: none !important;
        }

        .block-container {
            max-width: 1440px !important;
            padding-left: clamp(1rem, 4vw, 3rem) !important;
            padding-right: clamp(1rem, 4vw, 3rem) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">◉</div>
            <div class="brand-title">Face Recognition</div>
            <div class="brand-subtitle">Computer Vision System</div>
        </div>

        <div class="profile-card">
            <div class="profile-row">
                <div class="profile-avatar">👤</div>
                <div>
                    <div class="profile-name">Face Admin</div>
                    <div class="profile-role">System Administrator</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("NAVIGATION")

    menu = st.radio(
        "Navigation",
        ["Dashboard", "Foto", "Deteksi"],
        format_func=lambda x: {
            "Dashboard": "⌂   Dashboard",
            "Foto": "▣   Registrasi Wajah",
            "Deteksi": "◉   Real-time Deteksi",
        }[x],
        label_visibility="collapsed"
    )

    st.markdown(
        """
        <div style="margin-top: 35px; padding: 0 4px; color:#8fa5c5;
                    font-size:10px; line-height:1.6;">
            FACE RECOGNITION SYSTEM<br>
            Local AI • SQLite • OpenCV
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# LOAD DATABASE
# ============================================================

faces_df = load_faces()


# ============================================================
# SIDEBAR TOGGLE
# ============================================================

# Tombol toggle sidebar:
# « = tutup/sembunyikan sidebar
# » = buka/tampilkan kembali sidebar
# Tombol berada di kanan atas dan selalu tersedia.
# « = tutup sidebar, » = buka sidebar.
toggle_spacer, toggle_col = st.columns([0.93, 0.07])

with toggle_col:
    if st.session_state.sidebar_open:
        clicked = st.button(
            "«",
            key="hide_sidebar",
            help="Sembunyikan sidebar",
            use_container_width=True
        )
    else:
        clicked = st.button(
            "»",
            key="show_sidebar",
            help="Tampilkan sidebar",
            use_container_width=True
        )

    if clicked:
        toggle_sidebar()
        st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

if menu == "Dashboard":

    total_faces = len(faces_df)
    unique_people = faces_df["name"].nunique() if not faces_df.empty else 0

    st.markdown(
        """
        <div class="topbar">
            <div>
                <div class="eyebrow">Overview</div>
                <div class="page-title">Dashboard</div>
                <div class="page-subtitle">
                    Monitor data wajah dan aktivitas sistem pengenalan wajah.
                </div>
            </div>
            <div class="status-pill">
                <span class="status-dot"></span>
                System Online
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Statistic cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-icon">◉</div>
                <div class="stat-label">TOTAL DATA WAJAH</div>
                <div class="stat-value">{total_faces}</div>
                <div class="stat-caption">Foto tersimpan di database</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-icon">👥</div>
                <div class="stat-label">TOTAL ORANG</div>
                <div class="stat-value">{unique_people}</div>
                <div class="stat-caption">Identitas terdaftar</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-icon">✓</div>
                <div class="stat-label">DATABASE</div>
                <div class="stat-value">Ready</div>
                <div class="stat-caption">SQLite database aktif</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-icon">AI</div>
                <div class="stat-label">RECOGNITION</div>
                <div class="stat-value">HOG</div>
                <div class="stat-caption">Face detection engine</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # ========================================================
    # DASHBOARD LOWER SECTION
    # ========================================================

    left, right = st.columns([1.7, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Statistik Registrasi</div>
                <div class="panel-subtitle">
                    Jumlah data wajah berdasarkan bulan registrasi
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Data statistik bulanan.
        if faces_df.empty:
            chart_df = pd.DataFrame(
                {"Registrasi": [0] * 12},
                index=[
                    "Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                    "Jul", "Agu", "Sep", "Okt", "Nov", "Des"
                ]
            )
        else:
            dates = pd.to_datetime(
                faces_df["created_at"],
                errors="coerce"
            )

            monthly = (
                dates.dt.month
                .value_counts()
                .reindex(range(1, 13), fill_value=0)
            )

            chart_df = pd.DataFrame(
                {"Registrasi": monthly.values},
                index=[
                    "Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                    "Jul", "Agu", "Sep", "Okt", "Nov", "Des"
                ]
            )

        # Native Streamlit chart.
        # Ini sengaja digunakan agar HTML chart tidak berubah
        # menjadi blok kode pada browser.
        st.bar_chart(
            chart_df,
            height=270,
            use_container_width=True
        )

    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Wajah Terbaru</div>
                <div class="panel-subtitle">
                    Lima registrasi terakhir
                </div>
            """,
            unsafe_allow_html=True
        )

        if faces_df.empty:
            st.markdown(
                '<div style="color:#7B8798;font-size:12px;padding:20px 0;">'
                'Belum ada data wajah.</div></div>',
                unsafe_allow_html=True
            )
        else:
            recent = faces_df.head(5)

            html = ""

            for _, row in recent.iterrows():
                initial = (
                    str(row["name"]).strip()[:1].upper()
                    or "?"
                )

                html += f"""
<div class="recent-item">
    <div class="face-avatar">{initial}</div>
    <div>
        <div class="recent-name">{row["name"]}</div>
        <div class="recent-meta">
            {row["age"]} tahun • {row["created_at"]}
        </div>
    </div>
</div>
"""

            st.markdown(
                html + "</div>",
                unsafe_allow_html=True
            )

    st.write("")

    # Recent table
    st.markdown(
        """
        <div class="panel-title" style="margin: 5px 0 12px;">
            Data Wajah Terdaftar
        </div>
        """,
        unsafe_allow_html=True
    )

    if faces_df.empty:
        st.info("Belum ada data wajah yang terdaftar.")
    else:
        display_df = faces_df[
            ["id", "name", "age", "created_at"]
        ].copy()

        display_df.columns = [
            "ID", "Nama", "Umur", "Tanggal Registrasi"
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOTO / REGISTRATION
# ============================================================

elif menu == "Foto":

    st.markdown(
        """
        <div class="eyebrow">Registration</div>
        <div class="page-title">Registrasi Wajah</div>
        <div class="page-subtitle">
            Tambahkan identitas dan foto wajah baru ke dalam sistem.
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Data Personal</div>
                <div class="panel-subtitle">
                    Informasi pemilik data wajah
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        name = st.text_input(
            "Nama Lengkap",
            placeholder="Contoh: Fadly Febro"
        )

        age = st.number_input(
            "Umur",
            min_value=1,
            max_value=120,
            value=20,
            step=1
        )

        st.caption(
            "Gunakan nama dan umur yang sesuai dengan identitas "
            "orang pada foto."
        )

    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Foto Wajah</div>
                <div class="panel-subtitle">
                    Pastikan hanya satu wajah yang terlihat jelas.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        input_method = st.radio(
            "Metode pengambilan",
            ["Upload Foto", "Kamera"],
            horizontal=True,
            label_visibility="collapsed"
        )

        uploaded_image = None

        if input_method == "Upload Foto":
            uploaded_image = st.file_uploader(
                "Pilih foto",
                type=["jpg", "jpeg", "png"],
                help="Format JPG, JPEG, atau PNG."
            )
        else:
            uploaded_image = st.camera_input(
                "Ambil foto menggunakan kamera"
            )

    if uploaded_image:

        image = Image.open(uploaded_image).convert("RGB")

        st.write("")

        preview_col, action_col = st.columns([1, 1])

        with preview_col:
            st.markdown(
                '<div class="panel-title" style="margin-bottom:10px;">'
                'Preview</div>',
                unsafe_allow_html=True
            )
            st.image(
                image,
                use_container_width=True
            )

        with action_col:
            st.markdown(
                '<div class="panel-title" style="margin-bottom:10px;">'
                'Validasi & Simpan</div>',
                unsafe_allow_html=True
            )

            st.info(
                "Sistem akan mendeteksi wajah terlebih dahulu. "
                "Foto hanya dapat disimpan jika tepat satu wajah ditemukan."
            )

            if st.button(
                "Simpan Wajah",
                type="primary",
                use_container_width=True
            ):

                if not name.strip():
                    st.error("Nama wajib diisi.")

                else:
                    with st.spinner("Menganalisis wajah..."):
                        encoding, error = generate_face_encoding(image)

                    if error:
                        st.error(error)

                    else:
                        save_face(
                            name=name.strip(),
                            age=age,
                            image=image,
                            encoding=encoding
                        )

                        st.success(
                            f"Wajah {name.strip()} berhasil terdaftar."
                        )

                        st.rerun()


# ============================================================
# DETECTION
# ============================================================

elif menu == "Deteksi":

    st.markdown(
        """
        <div class="eyebrow">Computer Vision</div>
        <div class="page-title">Real-time Recognition</div>
        <div class="page-subtitle">
            Kamera tetap aktif untuk mendeteksi wajah secara langsung.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Camera status guide.
    # IMPORTANT: jangan beri baris kosong di tengah HTML,
    # karena Markdown Streamlit dapat menganggap bagian setelahnya
    # sebagai code block.
    st.markdown(
        """<div class="camera-status-guide">
<div class="guide-item">
<span class="guide-dot waiting"></span>
<div>
<strong>Tidak ada wajah</strong>
<small>Wajah belum terdeteksi pada kamera</small>
</div>
</div>
<div class="guide-item">
<span class="guide-dot detected"></span>
<div>
<strong>Wajah terdeteksi</strong>
<small>Sistem sedang melakukan pencocokan</small>
</div>
</div>
<div class="guide-item">
<span class="guide-dot recognized"></span>
<div>
<strong>Wajah dikenali</strong>
<small>Nama dan umur tampil pada bounding box</small>
</div>
</div>
</div>""",
        unsafe_allow_html=True
    )

    # Camera is ALWAYS displayed.
    # Database boleh kosong: processor akan menampilkan
    # "Tidak ada wajah terdeteksi" atau "Wajah terdeteksi • Tidak dikenali".
    webrtc_streamer(
        key="face-recognition",
        video_processor_factory=FaceRecognitionProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280, "max": 1280},
                "height": {"ideal": 720, "max": 720},
                "aspectRatio": {"ideal": 16 / 9},
                "facingMode": "user",
                "frameRate": {"ideal": 30, "max": 30}
            },
            "audio": False
        },
        async_processing=True
    )

    st.markdown(
            """
<div class="camera-note">
    <strong>Status recognition:</strong><br>
    • Tidak ada wajah di depan kamera →
      <b>Tidak ada wajah terdeteksi</b><br>
    • Ada wajah tetapi belum terdaftar →
      <b>Wajah terdeteksi • Tidak dikenali</b><br>
    • Wajah cocok dengan database →
      <b>Wajah dikenali • Nama • Umur</b>
</div>
            """,
            unsafe_allow_html=True
        )

    total_faces = len(faces_df)

    if total_faces == 0:
        st.caption(
            "Database belum memiliki wajah terdaftar. Kamera tetap dapat "
            "digunakan untuk menguji deteksi wajah. Registrasikan wajah "
            "terlebih dahulu jika ingin mengaktifkan pengenalan nama dan umur."
        )
    else:
        st.caption(
            f"{total_faces} data wajah tersedia untuk proses recognition."
        )

    st.caption(
        "Kamera berjalan kontinu; proses recognition berjalan di background "
        "agar video tidak menunggu proses AI. Gunakan pencahayaan yang cukup "
        "dan posisikan wajah menghadap kamera."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong>Face Recognition Dashboard</strong>
        &nbsp;•&nbsp;
        Python · Streamlit · OpenCV · face_recognition
        <br>
        Local Computer Vision System
    </div>
    """,
    unsafe_allow_html=True
)