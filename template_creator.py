import mss
import numpy as np
import cv2


MONITOR = {"top": 220, "left": 1200, "width": 120, "height": 150}
def processing():
    with mss.mss() as sct:
        #while True:
            screenshot = sct.grab(MONITOR)
            img = np.array(screenshot)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            edges = cv2.Canny(gray, 100, 200)

            #cv2.imshow('Processed', edges)

            #q to exit loop
           # if cv2.waitKey(25) & 0xFF == ord('q'):
                    #break


            cv2.imwrite('Templates/up_arrow.png', edges)

   # cv2.destroyAllWindows()
