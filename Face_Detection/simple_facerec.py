import cv2
import face_recognition
import os
import tensorflow as tf
img = cv2.imread("C:/Users/vcd09/PycharmProjects/chatbot/pythonProject/Ronaldo..jfif")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img2 = cv2.imread("C:/Users/vcd09/PycharmProjects/chatbot/pythonProject/images/Ronaldo.jpg")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]


# So sánh hai mã nhận dạng khuôn mặt
result = face_recognition.compare_faces([img_encoding], img_encoding2)

# In kết quả
print("Kết quả: ", result)

cv2.imshow('img',img)
cv2.imshow('img2',img2)
cv2.waitKey(0)


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def load_encoding_images(self, images_path):
        for filename in os.listdir(images_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image = face_recognition.load_image_file(os.path.join(images_path, filename))
                face_encoding = face_recognition.face_encodings(image)[0]
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(os.path.splitext(filename)[0])

    def detect_known_faces(self, frame):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []
        face_distances = []  # To store similarity distances

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            face_distance = face_recognition.face_distance(self.known_face_encodings, face_encoding)  # Calculate distances

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]

            face_names.append(name)
            face_distances.append(face_distance)  # Store distances

        return face_locations, face_names, face_distances



    # Define layers for Proposal Network (P-Net)
    def create_pnet():
        inputs = tf.keras.Input(shape=(None, None, 3))
        x = tf.keras.layers.Conv2D(10, kernel_size=(3, 3), activation='relu', padding='valid')(inputs)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        # Add more layers as needed
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    # Define layers for Refine Network (R-Net)
    def create_rnet():
        inputs = tf.keras.Input(shape=(None, None, 3))
        x = tf.keras.layers.Conv2D(20, kernel_size=(3, 3), activation='relu', padding='valid')(inputs)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        # Add more layers as needed
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    # Define layers for Output Network (O-Net)
    def create_onet():
        inputs = tf.keras.Input(shape=(None, None, 3))
        x = tf.keras.layers.Conv2D(30, kernel_size=(3, 3), activation='relu', padding='valid')(inputs)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        # Add more layers as needed
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model




