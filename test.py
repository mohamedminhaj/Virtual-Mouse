import cv2
import numpy as np
import mediapipe as mp
import time
import tkinter as tk
from pynput.mouse import Controller, Button  # For mouse control

##########################

wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Initialize tkinter to get screen dimensions
root = tk.Tk()

wScr, hScr = root.winfo_screenwidth(), root.winfo_screenheight()
root.withdraw()  # Hide the tkinter window as it's not needed for display

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize mouse Controller from pynput
mouse = Controller()

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

while True:
    # 1. Capture frame-by-frame
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # 2. Find hand landmarks
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Get landmark positions for index and middle fingers
            lmList = [(id, int(lm.x * wCam), int(lm.y * hCam)) for id, lm in enumerate(handLms.landmark)]
            x1, y1 = lmList[8][1:]  # Index finger tip
            x2, y2 = lmList[12][1:]  # Middle finger tip

            # 3. Check which fingers are up
            fingers = []
            if lmList[8][2] < lmList[6][2]:  # Index finger
                fingers.append(1)
            else:
                fingers.append(0)

            if lmList[12][2] < lmList[10][2]:  # Middle finger
                fingers.append(1)
            else:
                fingers.append(0)

            # 4. Only Index Finger : Moving Mode
            if fingers[0] == 1 and fingers[1] == 0:
                # 5. Convert Coordinates
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                # 6. Smoothen Values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # 7. Move Mouse
                mouse.position = (wScr - clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # 8. Both Index and middle fingers are up : Clicking Mode
            if fingers[0] == 1 and fingers[1] == 1:
                # 9. Find distance between fingers
                length = np.hypot(x2 - x1, y2 - y1)
                # 10. Click mouse if distance is short
                if length < 40:
                    cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 15, (0, 255, 0), cv2.FILLED)
                    mouse.click(Button.left, 1)

    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
