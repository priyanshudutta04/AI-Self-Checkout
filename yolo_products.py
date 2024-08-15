import cv2
from ultralytics import YOLO
import cvzone
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import keras
from keras.layers import TFSMLayer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from PIL import Image

classes = {
    'dove shampoo': 185,
    'lays': 10,
    'marble cake': 30,
    'maaza': 42,
    'munch': 5,
    'thums up': 50,
    'timepass biscuit': 25
}

items = []
prediction_counts = {class_name: 0 for class_name in classes}


classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup","fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli","carrot", "hot dog", 
    "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed","diningtable", "toilet",
    "tvmonitor", "laptop", "mouse", "remote", "keyboard", "mobile phone","microwave","oven","toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors","teddy bear", "hair drier", "toothbrush"
]

def imgProcess(image):
    target_size = (224, 224)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(image_rgb, target_size)
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  
    output = predict(img)
    
    return output

def predict(test_inp):
    products_model = tf.keras.Sequential([
    
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
        
        
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
        
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
        
        tf.keras.layers.Flatten(),
        
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    products_model.load_weights('custom_model.h5')

    prediction=products_model.predict(test_inp)
    return prediction



def make_square_with_padding(image):
    h, w, _ = image.shape
    max_side = max(h, w)
    
    square_img = np.ones((max_side, max_side, 3), dtype=np.uint8) * 255
    
    x_center = (max_side - w) // 2
    y_center = (max_side - h) // 2
    
    square_img[y_center:y_center+h, x_center:x_center+w] = image
    
    return square_img

def detect():

    total=0

    model=YOLO('yolov8n.pt')
    

    cap=cv2.VideoCapture(0)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    # fps = cap.get(cv2.CAP_PROP_FPS) 
    # out = cv2.VideoWriter('sample/demo2.avi', fourcc, fps, (1280, 720))   
    cap.set(3,1280)
    cap.set(4,720)

    while True:
        success, img=cap.read()
        results=model(img,stream=True)

        if not success:
            break

        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]                                
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)

                conf=round(float(box.conf[0]),2)                        
                id=int(box.cls[0])                                     
                class_name = classNames[id]

                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)         


                if class_name != "person":
                    cropped_img = img[y1:y2, x1:x2]
                    padded_img = make_square_with_padding(cropped_img)
                    predicted_pose = imgProcess(padded_img)

                    if np.max(predicted_pose)>0.9:
                        predicted_class_index = np.argmax(predicted_pose)
                        predicted_class = list(classes.keys())[predicted_class_index]
                        prediction_counts[predicted_class] += 1
                        

                        if prediction_counts[predicted_class] > 2 and predicted_class not in items:
                            items.append(predicted_class)
                            total += classes[predicted_class]
                

                        # cv2.imwrite('cropped_bottle.jpg', cropped_img)
                        cvzone.putTextRect(img,f'{predicted_class}',(max(0,x1),max(40,y1)))
                else:
                    cvzone.putTextRect(img,f'{class_name}',(max(0,x1),max(40,y1)))
                       
        
        height, width, _ = img.shape
        panel_width = 300  # Adjust the width as needed
        white_panel = np.ones((height, panel_width, 3), dtype=np.uint8) * 255

        text = "Invoice"
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 0.8
        font_thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = (panel_width - text_size[0]) // 2
        text_y = 40  

        
        item_font_scale = 0.6
        item_thickness = 1
        item_start_y = text_y + 40  # Starting Y position for list items
        item_spacing_y = 30  # Vertical spacing between items

        for i, item in enumerate(items):
            item_y = item_start_y + i * item_spacing_y
            item_x = 20  # Adjust the X position as needed
            item_value = classes[item]

            display_text = f'{item}: {item_value}'
            cv2.putText(white_panel, display_text, (item_x, item_y), font, item_font_scale, (0, 0, 0), item_thickness)



        cv2.putText(white_panel, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

        total_text = f'Total: {total}'
        total_font_scale = 0.7
        total_thickness = 1
        total_size, _ = cv2.getTextSize(total_text, font, total_font_scale, total_thickness)
        total_x = (panel_width - total_size[0]) // 2
        total_y = height - 40  # Positioning total at the bottom

        cv2.putText(white_panel, total_text, (total_x, total_y), font, total_font_scale, (0, 0, 0), total_thickness)

        img_with_panel = np.hstack((img, white_panel))

        # out.write(img_with_panel)
        cv2.imshow("Cam footage. Press 'Q' to exit.",img_with_panel)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            total = 0
            items.clear()  

    cap.release()
    # out.release()
    cv2.destroyAllWindows()

detect()
