import cv2
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load the labeled data from the CSV file
data = pd.read_csv('labels.csv')

# Vectorize the text data (if applicable)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Frame'].astype(str))
y = data['Label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = MultinomialNB()
a=model.fit(X_train, y_train)
tp=a.predict(X_train)
ac_train = accuracy_score(y_train,tp)

print(ac_train)

tp1=a.predict(X_test)
ac_test=accuracy_score(y_test,tp1)
print(ac_test)

# Function to label frames
def label_frame(frame):
    frame_text = vectorizer.transform([str(frame)])
    prediction = model.predict(frame_text)[0]
    return prediction

# Open the camera
cap = cv2.VideoCapture(0)

# Initialize variables for evaluation
predictions = []
true_labels = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform prediction on the current frame
    prediction = label_frame(frame)
    predictions.append(prediction)

    # Display the prediction on the frame
    cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Capture true labels (not necessary in real deployment)
    _, true_frame = cap.read()
    true_label = label_frame(true_frame)
    true_labels.append(true_label)

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

