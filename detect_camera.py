from yolo_detection_images import runModel
import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("images/1 - TensorFlow Sebastian Interview 01 V1.mp4")
    while True:
        ret, img = cap.read()
        cv2.imshow("Camera", runModel(img))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()