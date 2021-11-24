import cv2
import time
import HandTrackingModule as htm


def main():

    # Init the HandDetector
    detector = htm.handDetector()

    #Set video stream
    cap = cv2.VideoCapture(0)

    # Time for tracking fps
    pTime = 0
    cTime = 0
    # wImg, hImg  = 640, 480

    while True:

        # Get new image
        success, img = cap.read()

        img = detector.findHands(img, False)

        #Set target landmark in a given hand
        tgtLm = 8
        lmList = detector.findPosition(img,handNo=0, tgtLm=tgtLm ,
                                       drawCircleAroundLandMark=True)

        if len(lmList) != 0:
            print(lmList[tgtLm])
            # pass


        # Get the FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, "FPS: " + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255,0,255), 3 )

        cv2.imshow("Image", img)
        cv2.waitKey(1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
