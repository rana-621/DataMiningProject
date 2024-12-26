import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from transformers import pipeline
import tkinter as tk
from tkinter import filedialog


model = tf.keras.models.load_model('FinalBalance_model.h5')  # تأكد من أن المسار صحيح


def analyze_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))  # تغيير الحجم إلى (150, 150)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)
    probability = predictions[0][0]  # الحصول على الاحتمالية من النتيجة

    description = f"The model predicts with a probability of {probability:.2f}."

    
    if probability >= 0.5:  
        disease_type = "Pneumonia"
    else:  
        disease_type = "Normal"

    return description, disease_type


generator = pipeline("text-generation", model="gpt2")


def generate_medical_report(description, disease_type):
    if disease_type == "Normal":
        prompt = f"Generate a medical report based on the following analysis: {description}. The patient is healthy and no disease is found."
    else:
        prompt = f"Generate a medical report based on the following analysis: {description}. The patient shows signs of pneumonia, which may require medical attention and antibiotics."
    
    result = generator(prompt, max_length=150)
    return result[0]['generated_text']


def select_image():
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        description, disease_type = analyze_image(file_path)
        medical_report = generate_medical_report(description, disease_type)
        result_label.config(text=medical_report)


window = tk.Tk()
window.title("Medical Image Report Generator")


window.config(bg='#f0f0f0')  
window.geometry("600x500")  
window.resizable(False, False)  

title_label = tk.Label(window, text="Medical Image Report Generator", font=("Arial", 16, "bold"), bg='#f0f0f0', fg='#333')
title_label.pack(pady=10)

select_button = tk.Button(window, text="Select Image", font=("Arial", 12), bg='#4CAF50', fg='white', relief="solid", padx=20, pady=10, command=select_image)
select_button.pack(pady=20)


result_label = tk.Label(window, text="Generated report will appear here.", font=("Arial", 12), bg='#f0f0f0', fg='#333', wraplength=550, justify="left", anchor="w")
result_label.pack(pady=20, padx=10)


window.mainloop()
