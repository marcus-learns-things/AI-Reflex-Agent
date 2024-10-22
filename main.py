import pyautogui
import mss
import cv2
import time
import numpy as np

# Load templates
template_up = cv2.imread('./Templates/up_arrow.png', 0)
u_w, u_h = template_up.shape[::-1]

template_down = cv2.imread('./Templates/down_arrow.png', 0)
d_w, d_h = template_down.shape[::-1]

template_left = cv2.imread('./Templates/left_arrow.png', 0)
l_w, l_h = template_left.shape[::-1]

template_right = cv2.imread('./Templates/right_arrow.png', 0)
r_w, r_h = template_right.shape[::-1]

MONITOR = {"top": 290, "left": 960, "width": 600, "height": 600}

# Set a region where the note is considered to be "hit"
HIT_REGION_TOP = 2
HIT_REGION_BOTTOM = 40

# Cooldown period after pressing the note to avoid multiple hits
cooldown_time = 0.5

# Last hit time for each note to avoid hitting multiple times
last_hit_time = {
    'left': 0,
    'down': 0,
    'up': 0,
    'right': 0
}

# To store notes in hit region (avoid missing out)
hit_note_positions = {
    'left': False,
    'down': False,
    'up': False,
    'right': False
}

# Distance threshold for treating close matches as the same note
distance_threshold = 20

# Function to reset note positions after each hit
def reset_note_position(note_key):
    hit_note_positions[note_key] = False

# Function to detect and press the arrow keys with debouncing, cooldown logic, and note clearing
def detect_and_press(note_position, template_w, template_h, key_name, gray_frame, img, threshold=0.6):
    current_time = time.time()

    # Check if the note was recently pressed or still inside hit region
    if hit_note_positions[key_name] and current_time - last_hit_time[key_name] < cooldown_time:
        return None, None, None

    # Match the template to find the note
    res = cv2.matchTemplate(gray_frame, note_position, cv2.TM_SQDIFF_NORMED)
    min_val, _, min_loc, _ = cv2.minMaxLoc(res)

    # Lower values in TM_SQDIFF_NORMED indicate better matches
    if min_val <= threshold:
        top_left = min_loc
        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)

        # Draw a rectangle around detected note for debugging
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

        # Check if the note is within the hit region
        if HIT_REGION_TOP <= top_left[1] <= HIT_REGION_BOTTOM:
            return min_val, key_name, top_left

    return None, None, None

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

with mss.mss() as sct:
    while True:
        screenshot = sct.grab(MONITOR)
        img = np.array(screenshot)

        # Convert to grayscale for easier matching
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Draw the hit region for visual debugging
        cv2.line(img, (0, HIT_REGION_TOP), (MONITOR['width'], HIT_REGION_TOP), (0, 255, 255), 2)
        cv2.line(img, (0, HIT_REGION_BOTTOM), (MONITOR['width'], HIT_REGION_BOTTOM), (0, 255, 255), 2)

        # Store all detected notes in a list
        detected_notes = []

        # Detect notes and press corresponding keys with debouncing and cooldown logic
        for note_position, template_w, template_h, key_name in [
            (template_left, l_w, l_h, 'left'),
            (template_down, d_w, d_h, 'down'),
            (template_up, u_w, u_h, 'up'),
            (template_right, r_w, r_h, 'right')
        ]:
            match_val, key, pos = detect_and_press(note_position, template_w, template_h, key_name, gray, img)
            if match_val is not None:
                detected_notes.append((match_val, key, pos))

        # Sort detected notes by match value (lower values = better match)
        detected_notes.sort(key=lambda x: x[0])

        # Set to keep track of keys we've already pressed to avoid double pressing
        pressed_keys = set()

        # Check for conflicting notes (close to each other) and handle them
        for i, (match_val, key, pos) in enumerate(detected_notes):
            if hit_note_positions[key] or key in pressed_keys:
                continue  # Skip if this note is already being hit or has been pressed in this cycle

            # Check if any subsequent note is too close spatially
            for j in range(i + 1, len(detected_notes)):
                _, other_key, other_pos = detected_notes[j]
                if calculate_distance(pos, other_pos) < distance_threshold:
                    # Conflict detected, skip other note
                    print(f"Conflict: Skipping {other_key} at {other_pos}, too close to {key} at {pos}")
                    pressed_keys.add(other_key)  # Mark other note as pressed (conflict resolution)

            # Press the note if it's the best match
            pyautogui.press(key)
            print(f"Pressed {key} at {pos}")  # Debugging info
            last_hit_time[key] = time.time()  # Update last hit time
            hit_note_positions[key] = True  # Mark the note as hit
            pressed_keys.add(key)  # Track that this key has been pressed

        # Reset note positions when the notes exit hit region
        for key_name in ['left', 'down', 'up', 'right']:
            reset_note_position(key_name)

        # Show the processed image with debugging visuals
        cv2.imshow('Debug Screen', img)

        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
