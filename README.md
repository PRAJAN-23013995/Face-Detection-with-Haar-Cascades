# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows

# PROGRAM 
```
import numpy as np
import cv2 
import matplotlib.pyplot as plt
%matplotlib inline
model = cv2.imread('image_01.png',0)
withglass = cv2.imread('image_02.png',0)
group = cv2.imread('image_03.jpeg',0)
plt.imshow(model,cmap='gray')
plt.show()
plt.imshow(withglass,cmap='gray')
plt.show()
plt.imshow(group,cmap='gray')
plt.show()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detect_face(img):
    
  
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img) 
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
        
    return face_img
result = detect_face(withglass)
plt.imshow(result,cmap='gray')
plt.show()
result = detect_face(group)
plt.imshow(result,cmap='gray')
plt.show()
def adj_detect_face(img):
    
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2, minNeighbors=5) 
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
        
    return face_img

# Doesn't detect the side face.
result = adj_detect_face(group)
plt.imshow(result,cmap='gray')
plt.show()
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
def detect_eyes(img):
    
    face_img = img.copy()
  
    eyes = eye_cascade.detectMultiScale(face_img) 
    
    
    for (x,y,w,h) in eyes: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
        
    return face_img

result = detect_eyes(model)
plt.imshow(result,cmap='gray')
plt.show()
eyes = eye_cascade.detectMultiScale(withglass)
# White around the pupils is not distinct enough to detect Denis' eyes here!
result = detect_eyes(withglass)
plt.imshow(result,cmap='gray')
plt.show()
cap = cv2.VideoCapture(0)

# Set up matplotlib
plt.ion()
fig, ax = plt.subplots()

ret, frame = cap.read(0)
frame = detect_face(frame)
im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title('Video Face Detection')

while True:
    ret, frame = cap.read(0)

    frame = detect_face(frame)

    # Update matplotlib image
    im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.10)

   

cap.release()
plt.close()
```
# OUTPUT:

![b6f7ddda-6331-40cf-a5a7-01e0b1eea101](https://github.com/user-attachments/assets/ea0d2189-a550-4658-8c7b-3e1c1eae0e33)


![4dcb3194-254b-4c0a-8f5b-659abf6d6c76](https://github.com/user-attachments/assets/0718da0e-5866-4537-a347-a86ff61cafdb)

![e902c4fc-ff57-4435-abee-752bf20393e2](https://github.com/user-attachments/assets/3fcf2439-3ca3-4a49-8c43-2b3ba5e4fb2b)


![a681e376-495f-46d0-8cae-c95c2f20424d](https://github.com/user-attachments/assets/30869906-a064-4421-af0a-33b6147f402c)


![2ba93609-8ebd-4d42-bb7f-b41b6567ccae](https://github.com/user-attachments/assets/33e59507-3e7f-46b5-9d0c-8e5804b25f43)


![65eea25e-6ead-4e3f-995d-1139760dee06](https://github.com/user-attachments/assets/3a59a383-f0b6-4c17-be7e-d8d267d6ae84)

![3b2f9f4b-a4d1-4593-8ab6-dac31a85ff06](https://github.com/user-attachments/assets/e1598157-3f04-4057-8629-879a4f52ab3d)

![82c3bdbf-84e7-4a4d-8d50-2f8f0d598400](https://github.com/user-attachments/assets/70381ce6-5fe6-4dfa-8cfa-845f8fa23d40)

![Screenshot 2025-05-20 143657](https://github.com/user-attachments/assets/37c7f0ac-4862-4d61-9f1c-84c5ad1a0a8a)








