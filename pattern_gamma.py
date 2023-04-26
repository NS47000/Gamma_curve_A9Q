import cv2
import numpy as np
import os
import GlobalVar

#panel resolution
width=1440
height=1080
script_dir =  os.path.abspath(os.path.dirname(__file__))



def  generate_CrossLine_pattern(line_width):
    img=np.zeros((height,width),dtype=np.uint8)
    img[height//2-line_width:height//2+line_width,:]=255
    img[:,width//2-line_width:width//2+line_width]=255
    return img

def AddRoiPattern(img_in,size_ratio,shift_ration):
    hei,wid=img_in.shape[0:2]
    print("height: {}, width: {}".format(hei,wid))
    roi_size=hei*size_ratio/4
    for i in range(0,4):
        y=(hei//2)-shift_ration*hei//2+i//2*shift_ration*hei*2//2
        x=(wid//2)-shift_ration*wid//2+i%2*shift_ration*wid*2//2
        img_in[int(y-roi_size):int(y+roi_size),int(x-roi_size):int(x+roi_size)]=255

    img_out=img_in
    return img_out

def AddGrayPattern(img_in,size_ratio,shift_ration,gray_1,gray_2,gray_3,gray_4):
    hei,wid=img_in.shape[0:2]
    img_out=np.zeros(img_in.shape,dtype=np.uint8)
    print("height: {}, width: {}".format(hei,wid))
    roi_size=hei*size_ratio/4
    gray=[gray_1,gray_2,gray_3,gray_4]
    for i in range(0,4):
        y=(hei//2)-shift_ration*hei//2+i//2*shift_ration*hei*2//2
        x=(wid//2)-shift_ration*wid//2+i%2*shift_ration*wid*2//2
        img_out[int(y-roi_size):int(y+roi_size),int(x-roi_size):int(x+roi_size)]=gray[i]
    return img_out

def gamma_white(ratio):
    img=np.zeros((height,width),dtype=np.uint8)
    img[int(height//2-height*ratio//2):int(height//2+height*ratio//2),int(width//2-height*ratio//2):int(width//2+height*ratio//2)]=255
    return img
    
def gamma_haha(ratio):#漸層pattern
    img=np.zeros((height,width),dtype=np.uint8)
    for width_pos in range(int(width//2-height*ratio//2),int(width//2+height*ratio//2)):
        value=(width_pos-int(width//2-height*ratio//2))*255//(int(width//2+height*ratio//2)-int(width//2-height*ratio//2))
        img[int(height//2-height*ratio//2):int(height//2+height*ratio//2),width_pos]=value
    return img
    

    
    

if __name__ == '__main__':

    cross=generate_CrossLine_pattern(3)
    ROI=AddRoiPattern(cross,0.4,0.3)
    #IMG=gamma_haha(1)
    #cv2.imwrite(os.path.join(script_dir,"try.png"),IMG)
    cv2.imwrite(os.path.join(script_dir,"ROI.png"),ROI)
    
    

    


