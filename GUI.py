import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from datetime import datetime

# Load the pre-trained sign language detection model
model = load_model('sign_language_model.h5')  # Assuming this is your trained model

# Define known words to predict
class_names = ['Hello', 'Yes', 'No', 'Thank you', 'Goodbye']

# Time restriction: 6 PM to 10 PM
def is_time_allowed():
    current_time = datetime.now().time()
    return current_time >= datetime.strptime("18:00", "%H:%M").time() and current_time <= datetime.strptime("22:00", "%H:%M").time()

# Helper function to preprocess the image for the model
def preprocess_image(image, target_size=(64, 64)):
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict the sign language gesture from the image
def predict_sign_language(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

# Function to process the uploaded image
def process_image():
    if not is_time_allowed():
        messagebox.showerror("Error", "The model is only operational between 6 PM and 10 PM.")
        return
    
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        predicted_word = predict_sign_language(image)
        
        # Display the result in the GUI
        messagebox.showinfo("Prediction", f"Predicted Word: {predicted_word}")
        
        # Display the image in the GUI
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        
        label_image.configure(image=image_tk)
        label_image.image = image_tk

# Function to process real-time video for sign language detection
def process_video():
    if not is_time_allowed():
        messagebox.showerror("Error", "The model is only operational between 6 PM and 10 PM.")
        return
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict the sign language gesture
        predicted_word = predict_sign_language(frame)
        
        # Display the prediction on the video frame
        cv2.putText(frame, f"Prediction: {predicted_word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the video
        cv2.imshow('Sign Language Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Build the GUI with Tkinter
root = tk.Tk()
root.title("Sign Language Detection")

# Create buttons for image and video processing
btn_image = tk.Button(root, text="Upload Image", command=process_image)
btn_image.pack(pady=10)

btn_video = tk.Button(root, text="Real-Time Video", command=process_video)
btn_video.pack(pady=10)

# Label to show the uploaded image
label_image = tk.Label(root)
label_image.pack(pady=10)

# Start the GUI event loop
root.mainloop()
