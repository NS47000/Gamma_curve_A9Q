import cv2
import os
import time
def Q35_close():
    os.system("A9Q_GA2_show_pattern_YN_0614.bat E")
    
def Q35_showpattern(img):
    Q35_close()
    cv2.imwrite("ReadyToPush.png",img)
    time.sleep(0.5)
    os.system("python A9Q_show_image.py -R ReadyToPush.png -L ReadyToPush.png")
    time.sleep(0.5)
    


    

def Q35_remove_all():
    os.system("adb shell rm /storage/emulated/0/Pictures/*.png")

