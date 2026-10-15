import cv2
import time
import pygame
from ultralytics import YOLO

CAMERA_INDEX = 0

PHONE_CLASS_ID = 67     
PHONE_CONFIDENCE = 0.45

WARNING_TIME = 2.0        
ALARM_COOLDOWN = 3.0 

#calling AI
print("Loading AI model...")
model = YOLO("yolo11n.pt")   
print("AI model loaded.")

#adding audio for alerts 
pygame.mixer.init()
try: 
    pygame.mixer.music.load("warning.mp3")
except pygame.error:
    print("Warning: audio file not found.")
    print("The system will still detect distractions.")

#camera
camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

phone_detected_start = None
last_alarm_time = 0

phone_detected = False
phone_confidence = 0.0

def play_warning():
    global last_alarm_time
    current_time = time.time()

    #stop alarm from playing repeatedly 
    if current_time - last_alarm_time < ALARM_COOLDOWN:
        return

    print("WARNING: DRIVER DISTRACTED")

    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.play()
    except pygame.error:
        print("Audio file could not be played.")

    last_alarm_time = current_time

#main loop
while True:

    success, frame = camera.read()
    if not success:
        print("Could not read camera.")
        break

    #YOLO AI detection
    results = model(frame, verbose=False)
    #reset
    phone_detected = False
    phone_confidence = 0.0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            #checks if object is a phone 
            if class_id == PHONE_CLASS_ID and confidence >= PHONE_CONFIDENCE:
                phone_detected = True
                phone_confidence = max(phone_confidence, confidence)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                #draw phone box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"PHONE {confidence:.0%}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    #phone detection time
    current_time = time.time()

    if phone_detected:
        if phone_detected_start is None:
            phone_detected_start = current_time
        distraction_duration = current_time - phone_detected_start
    else:
        phone_detected_start = None
        distraction_duration = 0

    distraction_score = 0

    if phone_detected:
        distraction_score += 60

    if distraction_duration >= WARNING_TIME:
        distraction_score += 40

    distraction_score = min(distraction_score, 100)

    if distraction_score >= 80:
        status = "DANGER - DISTRACTION DETECTED"
        status_color = (0, 0, 255)
        play_warning()
    elif distraction_score >= 50:
        status = "WARNING"
        status_color = (0, 165, 255)
    else:
        status = "DRIVER FOCUSED"
        status_color = (0, 255, 0)

    # UI display
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (30, 30, 30), -1)
    cv2.putText(frame, "FOCUSDRIVE AI", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.putText(frame, f"Status: {status}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    y = 160
    cv2.putText(frame, f"Phone detected: {'YES' if phone_detected else 'NO'}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y += 40
    cv2.putText(frame, f"Phone confidence: {phone_confidence:.0%}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y += 40
    cv2.putText(frame, f"Distraction time: {distraction_duration:.1f}s",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y += 40
    cv2.putText(frame, f"Focus score: {100 - distraction_score}/100",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    bar_x = 20
    bar_y = y + 30
    bar_width = 400
    bar_height = 25

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_width, bar_y + bar_height),
                  (80, 80, 80), 2)

    filled_width = int(bar_width * distraction_score / 100)

    if filled_width > 0:
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + filled_width, bar_y + bar_height),
                      status_color, -1)
    #instructions 
    cv2.putText(frame, "Press Q to quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 2)

    cv2.imshow("FocusDrive AI", frame)  
    #exit 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("FocusDrive AI stopped.")
