import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

#Initialization:
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

#Audio Setup (Windows specific) (linux ka pata nahi, Rishi. i dont really use linux these days)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()     #[min, max, step]

#Canvas for Air Writing
canvas = None
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    if canvas is None: canvas = np.zeros_like(img)

    #Process Hand
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            #Get IDs for Thumb (4) and Index (8)
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                lm_list.append([int(lm.x * w), int(lm.y * h)])

            #1. VOLUME CONTROL (Thumb + Index distance)
            x1, y1 = lm_list[4]
            x2, y2 = lm_list[8]
            distance = np.hypot(x2 - x1, y2 - y1)
            vol = np.interp(distance, [30, 200], [vol_range[0], vol_range[1]])
            volume.SetMasterVolumeLevel(vol, None)

            # 2. AIR WRITING (Draw with Index finger)
            # If thumb is tucked in, it is treated as if the pen is down
            if distance < 40:
                cv2.circle(canvas, (x2, y2), 10, (0, 255, 0), cv2.FILLED)

            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    #Merge canvas and webcam feed
    img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)
    
    cv2.imshow("Hand Tracker - Press 'q' to exit", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()