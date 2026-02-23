import os
import cv2
import numpy as np
import urllib.request
import openvino as ov
import matplotlib.pyplot as plt
import collections
import time

# --- KONFIGURACJA ŚCIEŻEK (względem folderu projektu) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "intel")
BASE_URL = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/1/"
MODEL_NAMES = [
    "face-detection-retail-0004",
    "emotions-recognition-retail-0003",
    "head-pose-estimation-adas-0001",
]

FACE_XML = os.path.join(MODEL_DIR, "face-detection-retail-0004", "FP32", "face-detection-retail-0004.xml")
FACE_BIN = os.path.join(MODEL_DIR, "face-detection-retail-0004", "FP32", "face-detection-retail-0004.bin")
EMOTION_XML = os.path.join(MODEL_DIR, "emotions-recognition-retail-0003", "FP32", "emotions-recognition-retail-0003.xml")
EMOTION_BIN = os.path.join(MODEL_DIR, "emotions-recognition-retail-0003", "FP32", "emotions-recognition-retail-0003.bin")
HEAD_POSE_XML = os.path.join(MODEL_DIR, "head-pose-estimation-adas-0001", "FP32", "head-pose-estimation-adas-0001.xml")
HEAD_POSE_BIN = os.path.join(MODEL_DIR, "head-pose-estimation-adas-0001", "FP32", "head-pose-estimation-adas-0001.bin")

EMOTION_LABELS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Anger']


def download_models_if_missing():
    """Sprawdza obecność plików .xml i .bin w intel/<model>/FP32/. Jeśli brakuje — pobiera z Open Model Zoo."""
    for name in MODEL_NAMES:
        dir_fp32 = os.path.join(MODEL_DIR, name, "FP32")
        xml_path = os.path.join(dir_fp32, f"{name}.xml")
        bin_path = os.path.join(dir_fp32, f"{name}.bin")
        if os.path.isfile(xml_path) and os.path.isfile(bin_path):
            continue
        os.makedirs(dir_fp32, exist_ok=True)
        for ext in (".xml", ".bin"):
            url = BASE_URL + f"{name}/FP32/{name}{ext}"
            path = os.path.join(dir_fp32, f"{name}{ext}")
            if os.path.isfile(path):
                continue
            print(f"Pobieranie: {name}{ext} ...")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                raise RuntimeError(f"Nie udało się pobrać {url}: {e}")
    print("Modele gotowe.")


# --- ZMIENNE DO ANALIZY (PAMIĘĆ ASYSTENTA) ---
historia_yaw = collections.deque(maxlen=15)
historia_pitch = collections.deque(maxlen=15)
historia_emocji = collections.deque(maxlen=15)
raport_skupienia = []
czasy_pomiarow = []
start_czas = time.time()


def wygladz_dane(nowa_wartosc, bufor):
    bufor.append(nowa_wartosc)
    return sum(bufor) / len(bufor)


def stabilizuj_emocje(nowa_emocja, bufor):
    bufor.append(nowa_emocja)
    return collections.Counter(bufor).most_common(1)[0][0]


def ocen_skupienie(yaw, pitch, emocja):
    if abs(yaw) > 45:
        return "ROZPROSZONY (BOK)", 0
    elif abs(pitch) > 40:
        return "ROZPROSZONY (DOL/GORA)", 0
    elif emocja == "Sad":
        return "ZMECZENIE", 0.5
    else:
        return "SKUPIONY", 1


def main():
    download_models_if_missing()

    print("Ładowanie modeli...")
    core = ov.Core()
    face_net = core.compile_model(FACE_XML, "CPU")
    emotion_net = core.compile_model(EMOTION_XML, "CPU")
    head_pose_net = core.compile_model(HEAD_POSE_XML, "CPU")

    face_output = face_net.output(0)
    emotion_output = emotion_net.output(0)
    hp_out_y = head_pose_net.output("angle_y_fc")
    hp_out_p = head_pose_net.output("angle_p_fc")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Błąd: nie można otworzyć kamery (indeks 0).")
        return
    print("Start! Naciśnij 'q' aby zakończyć i zobaczyć wykres.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Wejście dla face-detection-retail-0004: 300x300, NCHW, float32
        img_input = cv2.resize(frame, (300, 300))
        img_input = img_input.transpose((2, 0, 1))  # HWC -> CHW
        img_input = np.expand_dims(img_input, axis=0).astype(np.float32)

        out = face_net([img_input])
        # Wynik ma kształt [1, 1, N, 7] (image_id, label, conf, x_min, y_min, x_max, y_max)
        results = out[face_output] if isinstance(out, dict) else out[0]
        detections = results.reshape(-1, 7)

        twarz_wykryta = False
        for detection in detections:
            confidence = float(detection[2])
            if confidence > 0.5:
                twarz_wykryta = True
                xmin = int(detection[3] * w)
                ymin = int(detection[4] * h)
                xmax = int(detection[5] * w)
                ymax = int(detection[6] * h)
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(w, xmax)
                ymax = min(h, ymax)

                face_crop = frame[ymin:ymax, xmin:xmax]
                if face_crop.size == 0:
                    continue

                face_hp = cv2.resize(face_crop, (60, 60))
                face_hp = face_hp.transpose(2, 0, 1)
                face_hp = np.expand_dims(face_hp, axis=0).astype(np.float32)
                hp_res = head_pose_net([face_hp])
                raw_yaw = hp_res[hp_out_y][0][0]
                raw_pitch = hp_res[hp_out_p][0][0]
                yaw = wygladz_dane(raw_yaw, historia_yaw)
                pitch = wygladz_dane(raw_pitch, historia_pitch)

                # KLOCEK 3: EMPATIA (emocje ze stabilizacją)
                face_em = cv2.resize(face_crop, (64, 64))
                face_em = face_em.transpose(2, 0, 1)
                face_em = np.expand_dims(face_em, axis=0).astype(np.float32)
                em_res = emotion_net([face_em])[emotion_output]
                nowa_emocja = EMOTION_LABELS[np.argmax(em_res)]
                emocja = stabilizuj_emocje(nowa_emocja, historia_emocji)

                status_tekst, punktacja = ocen_skupienie(yaw, pitch, emocja)
                raport_skupienia.append(punktacja)
                czasy_pomiarow.append(time.time() - start_czas)

                color = (0, 255, 0) if punktacja == 1 else (0, 0, 255)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, status_tekst, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Yaw: {int(yaw)} Pitch: {int(pitch)} Emocja: {emocja}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if not twarz_wykryta:
            raport_skupienia.append(0)
            czasy_pomiarow.append(time.time() - start_czas)
            cv2.putText(frame, "BRAK UCZNIA!", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Cyfrowy Trener Skupienia", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Generowanie raportu...")
    plt.figure(figsize=(10, 4))
    plt.plot(czasy_pomiarow, raport_skupienia, label='Poziom Skupienia')
    plt.axhline(y=0.8, color='r', linestyle='--', label='Próg sukcesu')
    plt.title("Raport Uważności Ucznia")
    plt.xlabel("Czas lekcji (sekundy)")
    plt.ylabel("Status (1=Skupiony, 0=Rozproszony)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
