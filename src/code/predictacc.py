import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the saved model
model = load_model('my_model.h5')

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)

# Function to preprocess the frame
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (64, 64))
    return np.expand_dims(resized_frame, axis=0)

# Function to label frames
def label_frame(frame):
    preprocessed_frame = preprocess_frame(frame)
    prediction = model.predict(preprocessed_frame)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# Lists to store ground truth and predicted labels
ground_truth_labels = []
predicted_labels = []

# Open the camera
cap = cv2.VideoCapture(0)

print("Press 'q' to stop capturing frames and calculate accuracy.")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform prediction on the current frame
    prediction = label_frame(frame)
    predicted_labels.append(prediction)

    # Display the prediction on the frame
    cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    frame_count += 1

    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Get ground truth labels from the user
for i in range(frame_count):
    ground_truth_label = input(f"Enter the ground truth label for frame {i + 1}/{frame_count}: ")
    ground_truth_labels.append(ground_truth_label)

# Ensure that the lengths of ground_truth_labels and predicted_labels match
if len(ground_truth_labels) != len(predicted_labels):
    print("Error: The number of ground truth labels does not match the number of predicted labels.")
else:
    # Calculate accuracy
    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
