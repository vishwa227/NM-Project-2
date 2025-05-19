import cv2
import numpy as np
import mediapipe as mp
import time
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import os

def choose_background():
    root = tk.Tk()
    root.withdraw()  # Hide the main Tk window
    messagebox.showinfo("Choose Background", "Select an image for your virtual background.")
    file_path = filedialog.askopenfilename(
        title="Select Background Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_path

def main():
    # GUI to select image
    bg_path = choose_background()
    if not bg_path or not os.path.exists(bg_path):
        print("No valid background selected. Exiting.")
        return

    # Load selected image
    bg_image = cv2.imread(bg_path)
    if bg_image is None:
        print("Error loading image.")
        return

    # Initialize MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    use_blur = False
    prev_time = time.time()

    print("Press 'q' to quit | Press 'b' to toggle blur mode.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Resize background to match webcam frame
        bg_resized = cv2.resize(bg_image, (w, h))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(rgb_frame)
        mask = results.segmentation_mask
        condition = np.stack((mask,) * 3, axis=-1) > 0.5

        if use_blur:
            blurred_bg = cv2.GaussianBlur(frame, (55, 55), 0)
            output_image = np.where(condition, frame, blurred_bg)
        else:
            output_image = np.where(condition, frame, bg_resized)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(output_image, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Virtual Background', output_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            use_blur = not use_blur

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()