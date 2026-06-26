from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import time
from collections import deque
import os
from datetime import datetime
import threading
import math


CAMERA_RESOLUTION = (640, 480)
CAMERA_FORMAT = "RGB888" 
CAFFE_MODEL_PATH = "mobilenet_iter_73000.caffemodel" 
CAFFE_PROTOTXT_PATH = "deploy.prototxt" 
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
PERSON_CLASS_NAME = "person" 
DNN_INPUT_SIZE = (224, 224) 
BLOB_SCALE_FACTOR = 0.007843
BLOB_MEAN_SUBTRACTION = (127.5, 127.5, 127.5)
DETECTION_THRESHOLD = 0.5
FRAME_SKIP_INTERVAL = 2 
FPS_AVERAGE_WINDOW_SIZE = 10
APP_PORT = 8000
TRACKER_TYPE = "MOSSE"


REQUIRED_CONSECUTIVE_DETECTIONS = 2 
REANCHOR_INTERVAL = 20              
MIN_COLOR_MATCH_SCORE = 0.5         
MAX_LOST_FRAMES = 150 
FINGERPRINT_UPDATE_INTERVAL = 15  

state_lock = threading.Lock()
current_mode = "DETECT" 
tracker = None
tracked_bbox = None
consecutive_detections = 0   
frames_in_track_mode = 0     
target_color_fingerprint = None 
lost_frames_counter = 0 
last_known_center = None 

ai_is_busy = False
ai_correction_bbox = None
last_console_msg = ""

# Inicjalizacja Kamery
picam2 = Picamera2()
try:
    picam2.configure(picam2.create_video_configuration(main={"size": CAMERA_RESOLUTION, "format": CAMERA_FORMAT}))
    picam2.start()
    print(f"Kamera gotowa: {CAMERA_RESOLUTION} w formacie {CAMERA_FORMAT}")
except Exception as e:
    print(f"Błąd inicjalizacji kamery: {e}"); exit()

# Inicjalizacja Modelu
try:
    net = cv2.dnn.readNetFromCaffe(CAFFE_PROTOTXT_PATH, CAFFE_MODEL_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
except Exception as e:
    print(f"Błąd ładowania modelu: {e}"); exit()

app = Flask(__name__)


def log_status(msg):
    """ Wypisuje komunikaty do konsoli bez spamowania """
    global last_console_msg
    if msg != last_console_msg:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        last_console_msg = msg

def extract_color_fingerprint(frame_rgb, bbox):
    x, y, w, h = [int(v) for v in bbox]
    img_h, img_w, _ = frame_rgb.shape
    x, y = max(0, x), max(0, y)
    w, h = min(img_w - x, w), min(img_h - y, h)
    if w <= 0 or h <= 0: return None
    
    roi = frame_rgb[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV) 
    hist = cv2.calcHist([hsv_roi], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def detect_objects_caffe_dnn(frame_rgb_input):
    frame_rgb = frame_rgb_input.copy()
    img_h, img_w, _ = frame_rgb.shape
    
    blob = cv2.dnn.blobFromImage(cv2.resize(frame_rgb, DNN_INPUT_SIZE), 
                                 BLOB_SCALE_FACTOR, DNN_INPUT_SIZE, BLOB_MEAN_SUBTRACTION,
                                 swapRB=True, crop=False) 
    net.setInput(blob)
    detections_output = net.forward()

    detected_items = []
    for i in range(detections_output.shape[2]):
        confidence = float(detections_output[0, 0, i, 2])
        class_id = int(detections_output[0, 0, i, 1])

        if 0 <= class_id < len(CLASSES) and CLASSES[class_id] == PERSON_CLASS_NAME and confidence > DETECTION_THRESHOLD:
            box = detections_output[0, 0, i, 3:7] * np.array([img_w, img_h, img_w, img_h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2)
            detected_items.append({
                "box": (x1, y1, x2, y2),
                "label": f"{CLASSES[class_id]}: {confidence:.2f}",
                "color": (255, 0, 0) 
            })
    return detected_items

def create_tracker(tracker_type):
    try: return cv2.legacy.TrackerMOSSE_create()
    except AttributeError: return cv2.TrackerMOSSE_create()


def background_ai_task(frame_rgb, target_fingerprint, previous_center):
    global ai_correction_bbox, ai_is_busy
    
    detected_items = detect_objects_caffe_dnn(frame_rgb)
    best_match_bbox = None
    highest_score = -1.0
    
    if target_fingerprint is not None:
        for det in detected_items:
            dnn_x1, dnn_y1, dnn_x2, dnn_y2 = det["box"]
            dnn_w, dnn_h = dnn_x2 - dnn_x1, dnn_y2 - dnn_y1
            
            if dnn_w > 10 and dnn_h > 10:
                candidate_bbox = (dnn_x1, dnn_y1, dnn_w, dnn_h)
                candidate_hist = extract_color_fingerprint(frame_rgb, candidate_bbox)
                
                if candidate_hist is not None:
                    score = cv2.compareHist(target_fingerprint, candidate_hist, cv2.HISTCMP_CORREL)
                    
                    if previous_center is not None:
                        cand_center_x = dnn_x1 + (dnn_w / 2)
                        cand_center_y = dnn_y1 + (dnn_h / 2)
                        dist = math.sqrt((cand_center_x - previous_center[0])**2 + (cand_center_y - previous_center[1])**2)
                        distance_penalty = min(dist / 300.0, 1.0) * 0.4
                        score = score - distance_penalty
                    
                    if score > highest_score and score > (MIN_COLOR_MATCH_SCORE - 0.1):
                        highest_score = score
                        best_match_bbox = candidate_bbox
    
    with state_lock:
        if best_match_bbox is not None:
            ai_correction_bbox = best_match_bbox
        ai_is_busy = False

def generate_frames():
    global current_mode, tracker, tracked_bbox, consecutive_detections, frames_in_track_mode
    global target_color_fingerprint, lost_frames_counter, ai_is_busy, ai_correction_bbox, last_known_center
    
    frame_count = 0
    fps_deque = deque(maxlen=FPS_AVERAGE_WINDOW_SIZE)
    last_processed_time_fps_calc = time.monotonic()
    last_detections_info = []
    

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 65]

    while True:
        try:

            raw_frame_rgb = picam2.capture_array() 
            raw_frame_rgb = cv2.cvtColor(raw_frame_rgb, cv2.COLOR_BGR2RGB)
        except Exception:
            time.sleep(0.1); continue

        frame_count += 1
        current_time_mono = time.monotonic()
        frame_to_encode = raw_frame_rgb.copy()
        
        with state_lock:
            mode_to_run = current_mode
            fingerprint_copy = target_color_fingerprint
        
        if mode_to_run == "DETECT":
            if fingerprint_copy is not None:
                lost_frames_counter += 1
                if lost_frames_counter > MAX_LOST_FRAMES:
                    with state_lock: target_color_fingerprint = None
                    lost_frames_counter = 0
                    log_status("TIMEOUT: Zgubiono cel definitywnie. Oczyszczam pamięć. Szukam nowych osób.")
            
            if FRAME_SKIP_INTERVAL > 1 and frame_count % FRAME_SKIP_INTERVAL != 0:
                for det in last_detections_info:
                    x1, y1, x2, y2 = det["box"]
                    cv2.rectangle(frame_to_encode, (x1, y1), (x2, y2), (150, 150, 150), 2)
            else:
                current_detections = detect_objects_caffe_dnn(raw_frame_rgb)
                last_detections_info = current_detections
                valid_candidates = []
                
                for det in current_detections:
                    x1, y1, x2, y2 = det["box"]
                    w, h = x2 - x1, y2 - y1
                    
                    if w > 15 and h > 20:
                        if fingerprint_copy is not None:
                            cand_hist = extract_color_fingerprint(raw_frame_rgb, (x1, y1, w, h))
                            if cand_hist is not None:
                                score = cv2.compareHist(fingerprint_copy, cand_hist, cv2.HISTCMP_CORREL)
                                
                                if last_known_center is not None:
                                    dist = math.sqrt(((x1 + w/2) - last_known_center[0])**2 + ((y1 + h/2) - last_known_center[1])**2)
                                    penalty = min(dist / 300.0, 1.0) * 0.4
                                    score = score - penalty
                                
                                if score > (MIN_COLOR_MATCH_SCORE - 0.1):
                                    det["score"] = score
                                    valid_candidates.append(det)
                                    cv2.rectangle(frame_to_encode, (x1, y1), (x2, y2), (255, 255, 0), 2) 
                                else:
                                    cv2.rectangle(frame_to_encode, (x1, y1), (x2, y2), (100, 100, 100), 1)
                        else:
                            valid_candidates.append(det)
                            cv2.rectangle(frame_to_encode, (x1, y1), (x2, y2), det["color"], 2)

                if valid_candidates:
                    if fingerprint_copy is not None:
                        valid_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
                        log_status(f"RECOVERY: Próba odzyskania celu... ({consecutive_detections+1}/{REQUIRED_CONSECUTIVE_DETECTIONS})")
                    else:
                        log_status(f"WERYFIKACJA: Widzę cel... ({consecutive_detections+1}/{REQUIRED_CONSECUTIVE_DETECTIONS})")
                        
                    first_person = valid_candidates[0]
                    x1, y1, x2, y2 = first_person["box"]
                    w, h = x2 - x1, y2 - y1
                    consecutive_detections += 1
                    
                    if consecutive_detections >= REQUIRED_CONSECUTIVE_DETECTIONS:
                        margin_w, margin_h = int(w * 0.15), int(h * 0.15)
                        init_bbox = (int(x1 + margin_w), int(y1 + margin_h), int(w - 2 * margin_w), int(h - 2 * margin_h)) 
                        
                        with state_lock:
                            if target_color_fingerprint is None:
                                target_color_fingerprint = extract_color_fingerprint(raw_frame_rgb, init_bbox)
                                log_status(">>> ZABLOKOWANO NOWY CEL! Zapisano profil. <<<")
                            else:
                                log_status(">>> ODZYSKANO ZGUBIONY CEL! Wznawiam śledzenie. <<<")
                                
                            tracked_bbox = init_bbox
                            last_known_center = (init_bbox[0] + init_bbox[2]/2, init_bbox[1] + init_bbox[3]/2)
                            tracker = create_tracker(TRACKER_TYPE)
                            tracker.init(raw_frame_rgb, tracked_bbox)
                            current_mode = "TRACK"
                            consecutive_detections = 0 
                            frames_in_track_mode = 0
                            lost_frames_counter = 0
                else:
                    consecutive_detections = 0
                    if fingerprint_copy is None:
                        log_status("Oczekuję na cel...")

        elif mode_to_run == "TRACK":
            frames_in_track_mode += 1
            
            with state_lock:
                if ai_correction_bbox is not None:
                    dnn_x, dnn_y, dnn_w, dnn_h = ai_correction_bbox
                    tracked_bbox = (int(dnn_x), int(dnn_y), int(dnn_w), int(dnn_h))
                    tracker = create_tracker(TRACKER_TYPE)
                    tracker.init(raw_frame_rgb, tracked_bbox)
                    ai_correction_bbox = None
                    # Fuksjowy w RGB: R=255, G=0, B=255
                    cv2.putText(frame_to_encode, "AI-SYNC", (int(tracked_bbox[0]), int(tracked_bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            success, current_bbox = tracker.update(raw_frame_rgb)
            
            if success:
                with state_lock:
                    old_x, old_y, old_w, old_h = tracked_bbox
                    new_x, new_y, new_w, new_h = current_bbox
                    smoothed_w = int((new_w * 0.7) + (old_w * 0.3))
                    smoothed_h = int((new_h * 0.7) + (old_h * 0.3))
                    tracked_bbox = (int(new_x), int(new_y), smoothed_w, smoothed_h)
                    
                    last_known_center = (new_x + smoothed_w/2, new_y + smoothed_h/2)
                
                x, y, w, h = tracked_bbox
                
                with state_lock:
                    if frames_in_track_mode % FINGERPRINT_UPDATE_INTERVAL == 0 and target_color_fingerprint is not None:
                        current_hist = extract_color_fingerprint(raw_frame_rgb, tracked_bbox)
                        if current_hist is not None:
                            target_color_fingerprint = (target_color_fingerprint * 0.8) + (current_hist * 0.2)

                    if frames_in_track_mode % REANCHOR_INTERVAL == 0 and fingerprint_copy is not None and not ai_is_busy:
                        ai_is_busy = True
                        threading.Thread(target=background_ai_task, args=(raw_frame_rgb.copy(), fingerprint_copy, last_known_center), daemon=True).start()

                if w > 0 and h > 0:

                    cv2.rectangle(frame_to_encode, (x, y), (x + w, y + h), (0, 255, 0), 2) 
                
                log_status("TRACKING: Cel w kadrze.")
            else:
                with state_lock:
                    current_mode = "DETECT"
                    tracker = None
                    consecutive_detections = 0
                    lost_frames_counter = 0
                log_status("!!! TRACKER ZGUBIŁ CEL !!! Przejście w tryb RECOVERY.")

        end_proc_time = time.monotonic()
        fps_val = 1.0 / (end_proc_time - last_processed_time_fps_calc) if (end_proc_time - last_processed_time_fps_calc) > 0 else 0
        last_processed_time_fps_calc = end_proc_time
        fps_deque.append(fps_val)
        if fps_deque: last_avg_fps = sum(fps_deque) / len(fps_deque)
        
        
        frame_bgr_for_encoding = cv2.cvtColor(frame_to_encode, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", frame_bgr_for_encoding, encode_params) 
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return f"""<html><head><title>Drone Vision Stream</title>
               <meta name="viewport" content="width=device-width, initial-scale=1.0">
               <style>
                    body {{ font-family: sans-serif; text-align: center; background-color: #222; color: #fff; margin: 0; padding: 10px; }} 
                    img {{ max-width: 100%; height: auto; border: 4px solid #444; border-radius: 8px; box-shadow: 0px 0px 15px rgba(0,0,0,0.5);}}
               </style>
               </head><body><h2>Dron: System Autonomicznego Śledzenia</h2>
               <img src="/video_feed"></body></html>"""

@app.route('/video_feed')
def video_feed_route(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print(f"Flask start: Otwórz przeglądarkę na adresie http://0.0.0.0:{APP_PORT}")
    print("================================================================")
    print("LOGI SYSTEMOWE DRONA:")
    try: app.run(host='0.0.0.0', port=APP_PORT, debug=False, threaded=True)
    except KeyboardInterrupt: pass
    finally:
        if 'picam2' in globals() and picam2.started: picam2.stop()
