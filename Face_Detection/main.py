import cv2
import face_recognition
from simple_facerec import SimpleFacerec
sfr = SimpleFacerec()
sfr.load_encoding_images("C:/Users/vcd09/PycharmProjects/chatbot/pythonProject/images/")

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect known faces
    face_locations, face_names, face_distances = sfr.detect_known_faces(frame)

    # Draw rectangles and names around faces
    for (top, right, bottom, left), name, face_distance in zip(face_locations, face_names, face_distances):
        # Display accuracy measure (rounded to two decimal places)
        accuracy =1- round(1 - face_distance[0], 2)  # Higher values indicate higher accuracy
        cv2.putText(frame, f"Accuracy: {accuracy:.2f}", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0,255), 1)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 200), 4)
        cv2.putText(frame, name, (left + 6, top - 10), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 200), 2)

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Press `q` to quit
    key = cv2.waitKey(1)
    if key == 27:
        break
    if key =='q':
        break

    # Example usage:
    # pnet = create_pnet()
    # rnet = create_rnet()
    # onet = create_onet()
# Release the video capture object
cap.release()

# Close all windows
cv2.destroyAllWindows()
