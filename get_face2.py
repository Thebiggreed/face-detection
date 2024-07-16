import cv2
import numpy as np
import os

# Load a set of face images of yourself and label them
images = []  # List to hold all images
labels = []  # List to hold all labels

def load_images(path, label):
    files = get_files(path, recursive=True)
    for image_path in files:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images.append(np.array(img, 'uint8'))
        labels.append(label)

def get_files(path, recursive=True):
    files = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            files.append(item_path)
        elif os.path.isdir(item_path) and recursive:
            files.extend(get_files(item_path, recursive))
    return files

def main():
    load_images("../face_detection/ME", 1)
    # load_images("MIA", 2)

    # Train the LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))

    capture = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier('File/haarcascade_frontalface_default.xml')

    # Initialize a counter for the statistics
    counter = {"Someone else": 0, "Michael": 0,"Mia": 0,}

    while True:
        _, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray)

        # Lists to store the faces and their confidence levels
        faces_and_confidences_michael = []
        faces_and_confidences_mia = []
        faces_and_confidences_someone_else = []

        for (x, y, w, h) in faces:
            label, confidence = recognizer.predict(gray[y: y + h, x: x + w])
            if label == 1:  # If the detected face is Michael
                faces_and_confidences_michael.append(((x, y, w, h), confidence))
            elif label == 2:  # If the detected face is Mia
                faces_and_confidences_mia.append(((x, y, w, h), confidence))
            else:
                faces_and_confidences_someone_else.append(((x,y,w,h,), confidence))

        # If faces were detected, select the face with the highest confidence level
        if faces_and_confidences_michael:
            # Sort the list by confidence levels in descending order and select the first element
            (x, y, w, h), confidence = sorted(faces_and_confidences_michael, key=lambda x: x[1], reverse=True)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Michael {round(confidence, 1)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if faces_and_confidences_mia:
            # Sort the list by confidence levels in descending order and select the first element
            (x, y, w, h), confidence = sorted(faces_and_confidences_mia, key=lambda x: x[1], reverse=True)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Mia {round(confidence, 1)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (182, 14, 255), 2)

        if faces_and_confidences_someone_else:
            # Sort the list
            (x, y, w, h), confidence = sorted(faces_and_confidences_someone_else, key=lambda x: x[1], reverse=True)[0]
            cv2.rectangle(frame (x,y), (x+w,y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Someone else {round(confidence, 1)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)


        cv2.imshow('face_detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
