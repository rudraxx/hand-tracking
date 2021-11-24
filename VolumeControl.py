import cv2
import time
import HandTrackingModule as htm
import math
import numpy as np

import argparse

def main(args):

    ############
    wImg, hImg = args.width , args.height

    # Dist between fingers
    minDist = args.lowerBound
    maxDist = args.upperBound
    videoDeviceNumber = args.videoDeviceNumber
    handType = args.handType
    flipImage = args.flipImage
    flagShowHands = args.showHands

    #VolumeBar related
    volBarWidth = 10
    volBarHeight = 200

    volBarX1 = 20
    volBarY1 = 130


    ############

    #Set video stream
    cap = cv2.VideoCapture(videoDeviceNumber)

    cap.set(3, wImg)
    cap.set(4, hImg)

    # Init the HandDetector
    detector = htm.handDetector(wImg= wImg, hImg = hImg, targetHandType=handType)

    # Time for tracking fps
    pTime = 0
    cTime = 0
    # wImg, hImg  = 640, 480

    while True:

        # Get new image
        success, img = cap.read()

        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        if(flipImage==1):
            img = cv2.flip(img, 1)

        img = detector.findHands(img, flagShowHands)

        #Set target landmark in a given hand
        tgtLmList = [4, 8] # Thumb, IndexFinger
        lmList = detector.findPosition(img, tgtLmList=tgtLmList ,
                                       drawCircleAroundLandMark=False)

        # Show the distance between in index finger and thumb
        distance = -1
        if len(lmList) != 0:

            #Get the position of the index and thumb fingers
            x1, y1, z1 = lmList[4][1],lmList[4][2], lmList[4][3]
            x2, y2, z2 = lmList[8][1], lmList[8][2], lmList[8][3]

            # print(x1,y1)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw markers on the image
            cv2.circle(img,(x1,y1), 10, (255,0,255), cv2.FILLED)
            cv2.circle(img,(x2, y2), 10, (255,0,255), cv2.FILLED)
            cv2.circle(img,(cx, cy), 10, (255,0,255), cv2.FILLED)
            cv2.line(img, (x1,y1),(x2,y2), (0,255,0), 2)

            #Calculate length
            length = math.hypot(float(x2-x1), float(y2-y1) )

            #Show as filled rectangle
            # Interp values between min and max
            calculatedVolume = np.interp(length, [minDist, maxDist], [0, volBarHeight])
            calculatedVolumePercent = np.interp(length, [minDist, maxDist], [0, 100])
            # print(vol)

            #SHow filled rectangle
            cv2.rectangle(img,(volBarX1,volBarY1+volBarHeight - int(calculatedVolume)), (volBarX1+volBarWidth,volBarY1+volBarHeight),(0,255,0), cv2.FILLED)
            cv2.putText(img, "Volume Percent: {:.2f}".format(calculatedVolumePercent) , (10, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255,0,255), 1 )

            #Show handedness of the detected Hand
            cv2.putText(img, "Hand: " + detector.detectedHandType , (10, 70), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255,0,255), 1 )


        # Show the FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, "FPS: " + str(int(fps)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255,0,255), 1 )


        # Show a rectangle as a volume bar
        cv2.rectangle(img,(volBarX1,volBarY1), (volBarX1+volBarWidth,volBarY1+volBarHeight),(0,255,0))

        # Show help to quit stream
        cv2.putText(img, "Press Esc to exit " , (wImg-200, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255,0,0), 1 )

        #Finally show the image
        cv2.imshow("Image", img)
        k = cv2.waitKey(1)
        if k==27: # Esc key to stop
            return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Volume Control Options")

    parser.add_argument("-wC", "--width",
                        type=int,
                        default=1280,
                        help="Width of Video Capture")
    parser.add_argument("-hC", "--height",
                        type=int,
                        default=960,
                        help="Height of Video Capture")

    parser.add_argument("-lb", "--lowerBound",
                        type=int,
                        default=10,
                        help="Lower bound for the distance between fingers")
    parser.add_argument("-ub", "--upperBound",
                        type=int,
                        default= 200,
                        help="Upper bound for the distance between fingers")
    parser.add_argument("-vd", "--videoDeviceNumber",
                        type=int,
                        default= 0,
                        help="videoDeviceNumber for opening camera stream")
    parser.add_argument("-f", "--flipImage",
                        type=int,
                        default= 1,
                        help="Flip image if left vs right hand seems inverted. ")
    parser.add_argument("-ht", "--handType",
                        type=str,
                        default= "Left",
                        help="Hand to be used for volume control")
    parser.add_argument("-sh", "--showHands",
                        type=int,
                        default= 1,
                        help="Show detected hand features")

    args = parser.parse_args()

    print("Using the following Arguments in the code:" , args)
    # print(args.upperBound)

    main(args)