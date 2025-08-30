import cv2
import mediapipe as mp
import pyautogui
import math
import time
import keyboard

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.85
)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

# Cursor smoothing buffer
buffer_size = 5
x_buffer = []
y_buffer = []

# Gesture cooldowns
cooldowns = {
    "click": 0.8,
    "drag": 1.2,
    "right_click": 1.0,
    "switch": 1.5,
    "panic": 2.0,
    "quit": 1.0,
    "back_forward": 1.0
}

last_used = {k: 0 for k in cooldowns}

dragging = False
drag_start = 0

# Swipe detection
prev_ix = None
prev_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    now = time.time()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            ix = int(lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame_w)
            iy = int(lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame_h)
            tx = int(lm[mp_hands.HandLandmark.THUMB_TIP].x * frame_w)
            ty = int(lm[mp_hands.HandLandmark.THUMB_TIP].y * frame_h)
            mx = int(lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * frame_w)
            my = int(lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * frame_h)

            screen_x = screen_w * lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            screen_y = screen_h * lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

            # Smoothing with buffer
            x_buffer.append(screen_x)
            y_buffer.append(screen_y)
            if len(x_buffer) > buffer_size:
                x_buffer.pop(0)
                y_buffer.pop(0)

            avg_x = sum(x_buffer) / len(x_buffer)
            avg_y = sum(y_buffer) / len(y_buffer)

            pyautogui.moveTo(avg_x, avg_y)

            # Pinch distances
            dist_index_thumb = math.hypot(ix - tx, iy - ty)
            dist_middle_thumb = math.hypot(mx - tx, my - ty)

            # Left click & drag
            if dist_index_thumb < 35:
                if now - last_used["click"] > cooldowns["click"]:
                    pyautogui.click()
                    last_used["click"] = now
                    drag_start = now
                elif now - drag_start > cooldowns["drag"] and not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            # Right click
            if dist_middle_thumb < 35 and now - last_used["right_click"] > cooldowns["right_click"]:
                pyautogui.rightClick()
                last_used["right_click"] = now

            # Hand signs
            def fingers_up(hand_landmarks):
                tips_ids = [4, 8, 12, 16, 20]
                fingers = []
                if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
                    fingers.append(1)
                else:
                    fingers.append(0)
                for id in range(1, 5):
                    if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                return fingers

            fingers = fingers_up(hand_landmarks)

            if fingers == [1,0,0,0,0] and now - last_used["switch"] > cooldowns["switch"]:
                pyautogui.hotkey('alt', 'tab')
                last_used["switch"] = now

            elif fingers == [0,0,0,0,0] and now - last_used["panic"] > cooldowns["panic"]:
                pyautogui.hotkey('win', 'd')
                last_used["panic"] = now

            elif fingers == [0,1,1,0,0] and now - last_used["panic"] > cooldowns["panic"]:
                pyautogui.hotkey('win', 'l')
                last_used["panic"] = now

            elif fingers == [0,0,0,0,1] and now - last_used["quit"] > cooldowns["quit"]:
                cap.release()
                cv2.destroyAllWindows()
                exit()

            # Swipe detection for back/forward
            if prev_ix is not None and prev_time is not None:
                delta_x = ix - prev_ix
                delta_t = now - prev_time
                if abs(delta_x) > 80 and delta_t < 0.3:
                    if delta_x > 0 and now - last_used["back_forward"] > cooldowns["back_forward"]:
                        keyboard.send("alt+right")
                        last_used["back_forward"] = now
                    elif delta_x < 0 and now - last_used["back_forward"] > cooldowns["back_forward"]:
                        keyboard.send("alt+left")
                        last_used["back_forward"] = now
                prev_ix = ix
                prev_time = now
            else:
                prev_ix = ix
                prev_time = now

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Jarvis Ultra", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
