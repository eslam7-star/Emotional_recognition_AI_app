import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import model_from_json
import pickle,joblib
import random
import pandas as pd

# Load the pickled model
loaded_model = joblib.load('model_2.pkl')

# Load the model and emotion dictionary
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("emotion_model.h5")

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

def map_emotion_encoding(emotion):
    encoding_map = {2: "Fear", 1: "Disgust", 4: "Neutral", 6: "Surprise", 0: "Angry", 5: "Sad", 3: "Happy"}
    return get_key_by_value(encoding_map,emotion)

def get_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  # If value is not found in the dictionary

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
        emotion_str = emotion_dict[maxindex]

        # Get values from dropdown menus
        sufficient_income = int(sufficient_income_combo.get())
        personal_awards = int(personal_awards_combo.get())
        time_for_passion = int(time_for_passion_combo.get())
        age = age_combo.get()
        gender = gender_combo.get()
        
        # Encode emotion
        encoded_emotion = map_emotion_encoding(emotion_str)

        # Prepare input data for prediction
        input_data = {
            'FRUITS_VEGGIES': random.randint(1, 10) ,
            'PLACES_VISITED': random.randint(1, 10),
            'CORE_CIRCLE': random.randint(1, 10) ,
            'SUPPORTING_OTHERS': random.randint(1, 10) ,
            'SOCIAL_NETWORK': random.randint(1, 10) ,
            'ACHIEVEMENT': random.randint(1, 10),
            'DONATION': random.randint(1, 10) ,
            'BMI_RANGE':  random.randint(1, 10) ,
            'TODO_COMPLETED':  random.randint(1, 10) ,
            'FLOW':  random.randint(1, 10) ,
            'DAILY_STEPS':  random.randint(1, 10) ,
            'LIVE_VISION':  random.randint(1, 10) ,
            'SLEEP_HOURS':  random.randint(1, 10) ,
            'LOST_VACATION':  random.randint(1, 10) ,
            'DAILY_SHOUTING':  random.randint(1, 10) ,
            'SUFFICIENT_INCOME': sufficient_income,
            'PERSONAL_AWARDS': personal_awards,
            'TIME_FOR_PASSION': time_for_passion,
            'Emotion': encoded_emotion,
            'AGE_21 to 35': age == "21 to 35",
            'AGE_36 to 50': age == "36 to 50",
            'AGE_51 or more': age == "51 or more",
            'AGE_Less than 20': age == "Less than 20",
            'GENDER_Female': gender == "Female",
            'GENDER_Male': gender == "Male"
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        print(input_df)
        # Predict using the loaded model
        predicted_value = loaded_model.predict(input_df)
        cv2.putText(frame, emotion_str, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the image with the emotion overlay
        cv2.imshow('Detected Emotion', frame)
        cv2.waitKey(0)

        # Close the window after a key is pressed
        cv2.destroyAllWindows()
        print("Predicted value:", predicted_value)
       
        cv2.putText(frame, emotion_str, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Close the window after a key is pressed
        cv2.destroyAllWindows()
        print("Predicted value:", predicted_value)
        output_window = tk.Toplevel()
        output_window.title("Prediction Results")
        output_window.geometry("500x300")

        # Create a text widget to display DataFrame and prediction
        output_text = tk.Text(output_window, wrap="word", font=("Helvetica", 12))
        output_text.pack(fill="both", expand=True)

        # Display DataFrame and prediction
        output_text.insert(tk.END, "Input DataFrame:\n")
        output_text.insert(tk.END, str(input_df))
        output_text.insert(tk.END, "\n\nPredicted value: " + str(predicted_value))



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
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+5), (0, 255, 0), 4)
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
title_label.pack(pady=5)

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
age_combo = create_dropdown("Select Age Range:", ["Less than 20", "21 to 35", "36 to 50", "51 or more"], input_frame)
gender_combo = create_dropdown("Select Gender:", ["Female", "Male"], input_frame)

button_frame = tk.Frame(root, bg='#f0f0f0')
button_frame.pack(pady=20)

upload_btn = tk.Button(button_frame, text="Upload an image", font=("Helvetica", 12, "bold"), command=upload_image, bg='#007BFF', fg='white')
upload_btn.pack(side=tk.LEFT, padx=5)
live_btn = tk.Button(button_frame, text="Live detection", font=("Helvetica", 12, "bold"), command=live_detection, bg='#28a745', fg='white')
live_btn.pack(side=tk.LEFT, padx=5)

panel = tk.Label(root, bg='#f0f0f0')
panel.pack(pady=20)

root.mainloop()
