from yolo_detection_images import runModel
import cv2

if __name__ == "__main__":
    # img = cv2.imread("images/tuanbn.jpg")
    img = cv2.imread("images/02Schildkrout-mediumSquareAt3X.jpg")
    reSize_Img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
    cv2.imshow("Detect Mask", runModel(reSize_Img))
    cv2.waitKey()
    cv2.destroyAllWindows()