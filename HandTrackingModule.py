import cv2
import mediapipe as  mp
import time

class handDetector():
    def __init__(self, static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 wImg= 640,
                 hImg = 480,
                 targetHandType = "Right"):

        self.static_image_mode=static_image_mode
        self.max_num_hands=max_num_hands
        self.model_complexity=model_complexity
        self.min_detection_confidence=min_detection_confidence
        self.min_tracking_confidence=min_tracking_confidence
        self.wImg = wImg
        self.hImg = hImg
        self.targetHandType = targetHandType


        self.mpHands = mp.solutions.hands

        self.hands = self.mpHands.Hands(self.static_image_mode,
                                        self.max_num_hands,
                                        self.model_complexity,
                                        self.min_detection_confidence,
                                        self.min_tracking_confidence)

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # print(type(handLms))
                    self.mpDraw.draw_landmarks(img,
                                               handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, tgtLmList = [4], drawCircleAroundLandMark=True):
        '''

        :param img:
        :param detectHandType: detectHandType Left or right
        :param tgtLmList: Where do you want to draw a circle
        :param drawCircleAroundLandMark:
        :return:
        '''

        lmList = []

        if self.results.multi_hand_landmarks:

            # Check if the correct hand is detected
            for idx, currentResult in enumerate(self.results.multi_handedness):
                detectedHandType = currentResult.classification[0].label
                if(detectedHandType==self.targetHandType):
                    self.detectedHandType = detectedHandType

                    #Which hand do you need it for
                    currentHand = self.results.multi_hand_landmarks[idx]

                    # print(type(self.results.multi_handedness[handNo].classification))
                    #Find the landmarks in this hand
                    # for handLms in self.results.multi_hand_landmarks:

                    for id, lm in enumerate(currentHand.landmark):
                        # lm x and y values in normalized coordinates.
                        # hImg, wImg, ch = img.shape
                        cx, cy, cz = int(lm.x * self.wImg), int(lm.y * self.hImg), int(lm.z * self.wImg)
                        # print(id, cx, cy, self.wImg, self.hImg)

                        lmList.append([id,cx,cy, cz])

                        # Draw circle at given landmark
                        if drawCircleAroundLandMark:
                            if id in tgtLmList:
                                cv2.circle(img,(cx,cy), 2, (255,0,0), cv2.FILLED)

        return lmList


def main():

    # Init the HandDetector
    detector = handDetector()

    #Set video stream
    cap = cv2.VideoCapture(0)

    # Time for tracking fps
    pTime = 0
    cTime = 0
    # wImg, hImg  = 640, 480

    while True:

        # Get new image
        success, img = cap.read()

        #Find Hands
        img = detector.findHands(img, True)

        #Set target landmark in a given hand
        tgtLmList = [4, 8]
        lmList = detector.findPosition(img,handNo=0, tgtLmList=tgtLmList ,
                                       drawCircleAroundLandMark=True)

        if len(lmList) != 0:
            for tgtLm in tgtLmList:
                print(lmList[tgtLm])

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
