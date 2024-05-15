import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import model_from_json

# Load the model and emotion dictionary
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("emotion_model.h5")

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Function to upload and process image
def upload_image():
    file_path = filedialog.askopenfilename()
    img = Image.open(file_path)
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img
    
    frame = cv2.imread(file_path)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        print("Emotion: ", emotion_dict[maxindex])
        print("Sufficient Income: ", sufficient_income_combo.get())
        print("Personal Awards: ", personal_awards_combo.get())
        print("Time for Passion: ", time_for_passion_combo.get())
        print("Week Meditation: ", week_meditation_combo.get())
        print("Age: ", age_combo.get())
        print("Gender: ", gender_combo.get())

# Function for live emotion detection using webcam
def live_detection():
    cap = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Live Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Setup GUI
root = tk.Tk()
root.title("Emotion Detection")
root.geometry("800x800")
root.configure(bg='#f0f0f0')

title_label = tk.Label(root, text="Emotion Detection", font=("Helvetica", 24, "bold"), bg='#f0f0f0')
title_label.pack(pady=10)

# Add dropdown lists
def create_dropdown(label_text, options, parent):
    frame = tk.Frame(parent, bg='#f0f0f0')
    frame.pack(pady=5)
    label = tk.Label(frame, text=label_text, font=("Helvetica", 12), bg='#f0f0f0')
    label.pack(side=tk.LEFT, padx=5)
    combo = ttk.Combobox(frame, values=options, font=("Helvetica", 12))
    combo.pack(side=tk.LEFT, padx=5)
    return combo

input_frame = tk.Frame(root, bg='#f0f0f0')
input_frame.pack(pady=20)

scale_options = [str(i) for i in range(1, 11)]
boolean_options = ["True", "False"]

sufficient_income_combo = create_dropdown("Sufficient Income (1-10):", scale_options, input_frame)
personal_awards_combo = create_dropdown("Personal Awards (1-10):", scale_options, input_frame)
time_for_passion_combo = create_dropdown("Time for Passion (1-10):", scale_options, input_frame)
week_meditation_combo = create_dropdown("Week Meditation (1-10):", scale_options, input_frame)
age_combo = create_dropdown("Select Age Range:", ["Less than 20", "21 to 35", "36 to 50", "51 or more"], input_frame)
gender_combo = create_dropdown("Select Gender:", ["Female", "Male"], input_frame)

button_frame = tk.Frame(root, bg='#f0f0f0')
button_frame.pack(pady=20)

upload_btn = tk.Button(button_frame, text="Upload an image", font=("Helvetica", 12, "bold"), command=upload_image, bg='#007BFF', fg='white')
upload_btn.pack(side=tk.LEFT, padx=10)
live_btn = tk.Button(button_frame, text="Live detection", font=("Helvetica", 12, "bold"), command=live_detection, bg='#28a745', fg='white')
live_btn.pack(side=tk.LEFT, padx=10)

panel = tk.Label(root, bg='#f0f0f0')
panel.pack(pady=20)

root.mainloop()
