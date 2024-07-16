import cv2
import numpy as np
import os

# Load a set of face images of yourself and label them
images = []  # List to hold all images
labels = []  # List to hold all labels

def load_images(path, label):
    for image_path in os.listdir(path):
        img = cv2.imread(os.path.join(path, image_path), cv2.IMREAD_GRAYSCALE)
        images.append(np.array(img, 'uint8'))
        labels.append(label)

def main():
    load_images("../face_detection/ME", 1)

    # Train the LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))

    capture = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier('File/haarcascade_frontalface_default.xml')

    while True:
        _, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            label, confidence = recognizer.predict(gray[y: y + h, x: x + w])
            if label == 1:  # If the detected face is Michael
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Michael {round(confidence, 1)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('face_detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()