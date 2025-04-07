import cv2
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the labeled data from the CSV file
data = pd.read_csv('labels.csv')

# Vectorize the text data (if applicable)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Frame'].astype(str))

# Initialize and train the model
model = MultinomialNB()
model.fit(X, data['Label'])

# Function to label frames
def label_frame(frame):
    frame_text = vectorizer.transform([str(frame)])
    prediction = model.predict(frame_text)[0]
    return prediction

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform prediction on the current frame
    prediction = label_frame(frame)

    # Display the prediction on the frame
    cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
