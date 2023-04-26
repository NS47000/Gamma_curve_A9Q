import cv2
import numpy as np
import pattern_gamma
from GlobalVar import *
import Flir_camera
import matplotlib.pyplot as plt
import time
import sys
import math

script_dir =  os.path.abspath(os.path.dirname(__file__))
threshold=30
gain=0
exp_255=111111
Needto_capture=True
size_ratio=0.5
shift_ration=0.4
def generate_ROI_pattern():
    cross=pattern_gamma.generate_CrossLine_pattern(3)
    ROI=pattern_gamma.AddRoiPattern(cross,size_ratio,shift_ration)
    return ROI
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )
def sortroi(element):
    return element[0]+element[2]+(element[1]+element[3])*2
    
def detect_roi(squares,ratio):
    ROIS=[]
    for square in squares:
        max_x=0
        max_y=0
        min_x=99999
        min_y=99999
        for point in square:
            if point[0]>max_x:
                max_x=point[0]
            if point[0]<min_x:
                min_x=point[0]
            if point[1]>max_y:
                max_y=point[1]
            if point[1]<min_y:
                min_y=point[1]
        mid_x=(min_x+max_x)/2
        mid_y=(min_y+max_y)/2
        length_x=max_x-min_x
        length_y=max_y-min_y
        shift_x=length_x*(1-ratio)/2
        shift_y=length_y*(1-ratio)/2
        ROIS.append([int(min_x+shift_x),int(min_y+shift_y),int(max_x-shift_x),int(max_y-shift_y)])
    print(ROIS)
    out=sorted(ROIS,key=sortroi)
    print(out)
    return out

def find_squares(img):
    squares = []
    img = cv2.GaussianBlur(img, (3, 3), 0)
    try:   
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        gray=img
    ret, bin = cv2.threshold(gray,threshold, 255, cv2.THRESH_BINARY)     
    cv2.namedWindow('My Image2', cv2.WINDOW_NORMAL)
    cv2.imshow('My Image2',bin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, _hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("轮廓数量：%d" % len(contours))
    index = 0
    # 轮廓遍历
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True) #计算轮廓周长
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True) #多边形逼近
        # 条件判断逼近边的数量是否为4，轮廓面积是否大于1000，检测轮廓是否为凸的
        print("Area:"+str(cv2.contourArea(cnt))+"cnt_len:"+str(len(cnt)))
        if  cv2.contourArea(cnt)>img.shape[0]*img.shape[1]//30:#and cv2.isContourConvex(cnt) and len(cnt) == 4 
            print("Area:"+str(cv2.contourArea(cnt))+"cnt_len:"+str(len(cnt)))
            M = cv2.moments(cnt) #计算轮廓的矩
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])#轮廓重心
            
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
            # 只检测矩形（cos90° = 0）
            #if max_cos < 0.3:
            # 检测四边形（不限定角度范围）
            if True:
                index = index + 1
                #cv2.putText(img,("#%d"%index),(cx,cy),font,0.7,(255,0,255),2)
                squares.append(cnt)
    return squares, img
def test_exp(ROI):
    gray_pattern=pattern_gamma.AddGrayPattern(ROI,size_ratio,shift_ration,63,127,191,255)
    cv2.imwrite(os.path.join(script_dir,"data","gray_pattern.png"),gray_pattern)
    Q35_showpattern(gray_pattern)
    roi_list1=[]
    roi_list2=[]
    roi_list3=[]
    roi_list4=[]
    exp_list=[]
    for exp in range(11111,333333,22222):
        Flir_camera.getpicture(os.path.join(script_dir,"data","gray_"+str(exp)+".png"),exp,gain)
        img=cv2.imread(os.path.join(script_dir,"data","gray_"+str(exp)+".png"),1)
        roi_1=img[ROIs[0][1]:ROIs[0][3],ROIs[0][0]:ROIs[0][2]]
        roi_2=img[ROIs[1][1]:ROIs[1][3],ROIs[1][0]:ROIs[1][2]]
        roi_3=img[ROIs[2][1]:ROIs[2][3],ROIs[2][0]:ROIs[2][2]]
        roi_4=img[ROIs[3][1]:ROIs[3][3],ROIs[3][0]:ROIs[3][2]]
        print(np.mean(np.mean(roi_1)))
        print(np.mean(np.mean(roi_2)))
        print(np.mean(np.mean(roi_3)))
        print(np.mean(np.mean(roi_4)))
        roi_list1.append(np.mean(np.mean(roi_1)))
        roi_list2.append(np.mean(np.mean(roi_2)))
        roi_list3.append(np.mean(np.mean(roi_3)))
        roi_list4.append(np.mean(np.mean(roi_4)))
        exp_list.append(exp)
    
    path = 'line.csv'
    with open(path,'w') as f:
        for i in range(0,len(exp_list)):
            f.write(str(exp_list[i])+","+str(roi_list1[i])+","+str(roi_list2[i])+","+str(roi_list3[i])+","+str(roi_list4[i])+"\n")
    plt.plot(exp_list, roi_list1, 'ro-', linewidth=3)
    plt.plot(exp_list, roi_list2, 'go-', linewidth=3) 
    plt.plot(exp_list, roi_list3, 'bo-', linewidth=3) 
    plt.plot(exp_list, roi_list4, 'co-', linewidth=3)  
    plt.legend(('gray63', 'gray127', 'gray191', 'gray255'), loc='upper left')
    plt.grid(True)
    plt.show()
    

if __name__ == '__main__':
    """
    white=pattern_gamma.gamma_white(0.6)    
    haha=pattern_gamma.gamma_haha(0.6) 
    Q35_showpattern(white)    
    Flir_camera.getpicture(os.path.join(script_dir,"data","ROI_2.png"),exp_255,gain)
    roi_img=cv2.imread(os.path.join(script_dir,"data","ROI_2.png"),0)
    Q35_showpattern(haha) 
    Flir_camera.getpicture(os.path.join(script_dir,"data","detect.png"),exp_255,gain)
    detect_img=cv2.imread(os.path.join(script_dir,"data","detect.png"),0)
    squares, img = find_squares(roi_img)
    print(squares)
    result=(detect_img*255.0/roi_img).astype(np.uint8)
    cv2.imwrite(os.path.join(script_dir,"data","result.png"),result)
    print(str(squares[0][3][1])+str(",")+str(squares[0][1][1])+str(",")+str(squares[0][0][0])+str(",")+str(squares[0][2][0]))
    result2=result[squares[0][3][1]:squares[0][1][1],squares[0][0][0]:squares[0][2][0]]
    cv2.imwrite(os.path.join(script_dir,"data","result_2.png"),result2)
    sys.exit(1)
    """
    Needto_capture=False
    #method1
    if Needto_capture==True:
        ROI=generate_ROI_pattern()
        Q35_showpattern(ROI)
        if os.path.exists(os.path.join(script_dir,"data"))!=True:
            os.mkdir(os.path.join(script_dir,"data"))
        
        Flir_camera.getpicture(os.path.join(script_dir,"data","ROI.png"),exp_255,gain)
    
    roi_img=cv2.imread(os.path.join(script_dir,"data","ROI.png"),1)
    squares, img = find_squares(roi_img)
    cv2.drawContours( roi_img, squares, -1, (0, 0, 255), 2 )
    cv2.imwrite(os.path.join(script_dir,"data","ROI_detect.png"),roi_img)
    ROIs=detect_roi(squares,0.9)
    #print(ROIs)
    #test_exp(ROI)
    #sys.exit(1)
    exp_base=exp_255
    exp=0
    data_detect=[]
    data_show=[]
    if Needto_capture==True:
        white_pattern=pattern_gamma.AddGrayPattern(ROI,size_ratio,shift_ration,255,255,255,255)
        Q35_showpattern(white_pattern)
        Flir_camera.getpicture(os.path.join(script_dir,"data","balance.png"),exp_255,gain)
    img_balance=cv2.imread(os.path.join(script_dir,"data","balance.png"),1)
    roi_1_balance=np.mean(np.mean(img_balance[ROIs[0][1]:ROIs[0][3],ROIs[0][0]:ROIs[0][2]]))
    roi_2_balance=np.mean(np.mean(img_balance[ROIs[1][1]:ROIs[1][3],ROIs[1][0]:ROIs[1][2]]))
    roi_3_balance=np.mean(np.mean(img_balance[ROIs[2][1]:ROIs[2][3],ROIs[2][0]:ROIs[2][2]]))
    roi_4_balance=np.mean(np.mean(img_balance[ROIs[3][1]:ROIs[3][3],ROIs[3][0]:ROIs[3][2]]))
    print("Balance:"+str(roi_1_balance)+","+str(roi_2_balance)+","+str(roi_3_balance)+","+str(roi_4_balance))
    if Needto_capture==True:
        black_pattern=pattern_gamma.AddGrayPattern(ROI,size_ratio,shift_ration,0,0,0,0)
        Q35_showpattern(black_pattern)
        Flir_camera.getpicture(os.path.join(script_dir,"data","black.png"),exp_255*10,gain)
    img_balance=cv2.imread(os.path.join(script_dir,"data","black.png"),1)
    roi_1_black=np.mean(np.mean(img_balance[ROIs[0][1]:ROIs[0][3],ROIs[0][0]:ROIs[0][2]]))/10
    roi_2_black=np.mean(np.mean(img_balance[ROIs[1][1]:ROIs[1][3],ROIs[1][0]:ROIs[1][2]]))/10
    roi_3_black=np.mean(np.mean(img_balance[ROIs[2][1]:ROIs[2][3],ROIs[2][0]:ROIs[2][2]]))/10
    roi_4_black=np.mean(np.mean(img_balance[ROIs[3][1]:ROIs[3][3],ROIs[3][0]:ROIs[3][2]]))/10
    print("black noise:"+str(roi_1_black)+","+str(roi_2_black)+","+str(roi_3_black)+","+str(roi_4_black))
    
    for grayLevel in range(0,255,4):
        if Needto_capture==True:
            ratio=0
            gray_pattern=pattern_gamma.AddGrayPattern(ROI,size_ratio,shift_ration,grayLevel,grayLevel+1,grayLevel+2,grayLevel+3)
            Q35_showpattern(gray_pattern)
        if grayLevel<64:
            ratio=4
        elif grayLevel<128:
            ratio=2.5
        elif grayLevel<191:
            ratio=1.5
        else:
            ratio=1
        exp=exp_base*ratio
        if Needto_capture==True:
            Flir_camera.getpicture(os.path.join(script_dir,"data","gray"+str(grayLevel)+"_"+str(exp)+".png"),exp,gain)
        img=cv2.imread(os.path.join(script_dir,"data","gray"+str(grayLevel)+"_"+str(exp)+".png"),1)
        roi_1=img[ROIs[0][1]:ROIs[0][3],ROIs[0][0]:ROIs[0][2]]
        roi_2=img[ROIs[1][1]:ROIs[1][3],ROIs[1][0]:ROIs[1][2]]
        roi_3=img[ROIs[2][1]:ROIs[2][3],ROIs[2][0]:ROIs[2][2]]
        roi_4=img[ROIs[3][1]:ROIs[3][3],ROIs[3][0]:ROIs[3][2]]
        data_show.append(grayLevel)
        data_show.append(grayLevel+1)
        data_show.append(grayLevel+2)
        data_show.append(grayLevel+3)
        data_detect.append((np.mean(np.mean(roi_1))+0-roi_1_black*ratio)/ratio*255/roi_1_balance)
        data_detect.append((np.mean(np.mean(roi_2))+0-roi_2_black*ratio)/ratio*255/roi_2_balance)
        data_detect.append((np.mean(np.mean(roi_3))+0-roi_3_black*ratio)/ratio*255/roi_3_balance)
        data_detect.append((np.mean(np.mean(roi_4))+0-roi_4_black*ratio)/ratio*255/roi_4_balance)
    fig=plt.figure()
    plt.title("Gamma curve ")
    
    data_show_2=np.zeros((len(data_show)//4),dtype=np.float32)
    data_detect_2=np.zeros((len(data_detect)//4),dtype=np.float32)
    gamma22=np.zeros((len(data_detect)//4),dtype=np.float32)
    for i in range(0,len(data_detect),4):
        data_show_2[i//4]=data_show[i]
        data_detect_2[i//4]=np.mean(data_detect[i:i+3])
        gamma22[i//4]=math.pow(data_show_2[i//4]/255,2.2)*255
        
    plt.plot(data_show_2, data_detect_2, 'go-', linewidth=3)
    plt.plot(data_show_2,gamma22, 'bo-', linewidth=3) 
    plt.legend(('measurement', 'gamma2.2'), loc='upper left')
    plt.grid(True)
    plt.show() 
    fig.savefig("Q35_Gamma_curve.png")
    path = 'result.csv'
    with open(path,'w') as f:
        for i in range(0,len(data_show)):
            f.write(str(data_show[i])+","+str(data_detect[i])+"\n")

        
        
        
            
    
    
    
    
    
    
    