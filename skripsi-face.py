import cv2
import dlib
import numpy as np
import os
import pickle

# --- Pengaturan awal ---
dataset_dir = "dataset"
encoding_file = "data_masuk.pkl"
os.makedirs(dataset_dir, exist_ok=True)

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# --- Fungsi simpan encoding wajah ---
def save_face_encoding(name, image):
    dets = detector(image, 1)
    if len(dets) == 0:
        return False

    shape = sp(image, dets[0])
    face_descriptor = facerec.compute_face_descriptor(image, shape)
    face_descriptor_np = np.array(face_descriptor)

    # Simpan encoding
    if os.path.exists(encoding_file):
        with open(encoding_file, 'rb') as f:
            data = pickle.load(f)
    else:
        data = {}

    data[name] = face_descriptor_np
    with open(encoding_file, 'wb') as f:
        pickle.dump(data, f)

    return True

# --- Fungsi muat semua wajah dikenal ---
def load_known_faces():
    if not os.path.exists(encoding_file):
        return {}, []
    with open(encoding_file, 'rb') as f:
        data = pickle.load(f)
    return data, list(data.keys())

# --- Fungsi cari nama wajah ---
def recognize_face(face_encoding, known_encodings):
    for name, known_encoding in known_encodings.items():
        dist = np.linalg.norm(face_encoding - known_encoding)
        if dist < 0.6:
            return name
    return "Tidak Dikenal"

# --- Main Program ---
cap = cv2.VideoCapture(0)
nama = input("Masukkan nama (untuk simpan wajah baru): ")
disimpan = False

known_encodings, known_names = load_known_faces()

print("[INFO] Tekan 's' untuk simpan wajah, atau tunggu deteksi otomatis.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(gray, 1)

    for det in dets:
        shape = sp(gray, det)
        face_encoding = np.array(facerec.compute_face_descriptor(gray, shape))
        name = recognize_face(face_encoding, known_encodings)

        # Gambar kotak & nama
        x1, y1, x2, y2 = det.left(), det.top(), det.right(), det.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Wajah Dikenali", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not disimpan:
        # Simpan wajah
        saved = save_face_encoding(nama, gray)
        if saved:
            print(f"[INFO] Wajah '{nama}' disimpan.")
            known_encodings, known_names = load_known_faces()
            disimpan = True
        else:
            print("[WARNING] Tidak ada wajah yang terdeteksi.")

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
#kalau close tekan tombol ESC