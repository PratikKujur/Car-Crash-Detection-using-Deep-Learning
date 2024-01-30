import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model('C:/Users/ABHISHEK BAGHEL/Desktop/25/model.h5')

# Create a function to preprocess the input frameQ
def preprocess_frame(frame):

    # Resize the frame to the size required by the model
    resized = cv2.resize(frame, (224, 224))
    # Convert the resized frame to a numpy array
    array = np.array(resized)
    # Reshape the array to have a single channel
    reshaped = array.reshape(1, 224, 224, 3)
    # Normalize the pixel values to be between 0 and 1
    normalized = reshaped / 255.0
    return normalized

# Create a function to perform classification on the input frame
def classify_frame(frame):
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    # Use the trained model to predict the class of the preprocessed frame
    prediction = model.predict(preprocessed_frame)
    if prediction[0] > 0.5:
        label = 'Not Crash'

    else:
        label = 'Crash'

    print(label)
    # Get the class with the highest predicted probability
    class_index = np.argmax(prediction)
    # Return the class label

    return class_index

# Open the video file
cap = cv2.VideoCapture('C://Users//ABHISHEK BAGHEL//Desktop//Major Project//dataset//1 (55).mp4')


# Loop through the frames in the video
while(cap.isOpened()):
    # Read the next frame from the video
    ret, frame = cap.read()
    if ret:
        # Perform classification on the current frame
        #class_label = classify_frame(frame)
        #print (class_label)
        preprocessed_frame = preprocess_frame(frame)
        # Use the trained model to predict the class of the preprocessed frame
        prediction = model.predict(preprocessed_frame)

        if prediction[0] > 0.35:
            label = 'Not Crash'
            cv2.putText(frame, str(label), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

        else:
            label = 'Crash'
            cv2.putText(frame, str(label), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255), 3, cv2.LINE_AA)

        # Display the class label on the frame
        #cv2.putText(frame, str(label), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        # Display the frame
        cv2.imshow('frame', frame)
        # Wait for a key press
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
