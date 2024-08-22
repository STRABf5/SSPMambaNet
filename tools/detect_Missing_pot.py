import os
import numpy as np
import cv2
def compute_IoU(rec1,rec2):
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
  
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
   
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)


def plantDIOU(se_label,fl_label):
    pair=[]
    pot_seedling_store = {i:[] for i in range(len(fl_label))} # key is pot index, value is the list of (seeding idx, diou, distance)
    for pot_i, fl_line in enumerate(fl_label):
        max_diou = 0 # 
        closest_pot = None  
        for seedling_i, se_line in enumerate(se_label):
            diou ,d2= compute_DIOU(fl_line[1:],se_line[1:])#
            if diou>-0.09:#diou>-0.2 
                pot_seedling_store[pot_i].append((seedling_i, diou, d2))
    
    full, lack_one, empty = [], [], []
    # seedling tuple format: [(seedling idx, diou, distance),]
    for pot_i, seedling_list  in pot_seedling_store.items():
        sz = len(seedling_list)
        if sz == 0:
            empty.append(pot_i)
        elif sz == 1:
            lack_one.append(pot_i)
        elif sz == 2:
            full.append(pot_i)
        else:  
            # second filter for others
            seedling_list = sorted(seedling_list, key=lambda x:x[1], reverse=True)
            seedling_list = seedling_list[:2]
            pot_seedling_store[pot_i] = seedling_list
            full.append(pot_i)

    return full, lack_one, empty, pot_seedling_store

def draw_matches_on_images_full(image, matched_pairs):#
    matches_line=[]
    fl=0
    se=0
    que=[]
    noque=[]
    distence=[]
    missing_plants = []
    for matches in matched_pairs:
        
        fl+=1
        flowerpot_coords = matches[0]#
        flowerpot_x_min, flowerpot_y_min, flowerpot_x_max, flowerpot_y_max = flowerpot_coords[1:]
        center = ((flowerpot_x_min+flowerpot_x_max)//2,(flowerpot_y_min+flowerpot_y_max)//2)
        radius = min((flowerpot_x_max-flowerpot_x_min)//2,(flowerpot_y_max-flowerpot_y_min)//2)
        cv2.circle(image, center, radius, (0,255,255), thickness=14)
        
        plant_coords = matches[1]#
        plant_x_min, plant_y_min, plant_x_max, plant_y_max = plant_coords[1:]
       
        cv2.rectangle(image, (plant_x_min, plant_y_min), (plant_x_max, plant_y_max), (0, 0,255), 14)
      
        cv2.rectangle(image, (0, 0), (400, 300), (255, 255, 255), -1)  # 创建一个白色的矩形
   
    fl_full=str(int(fl/2))
    #fl= 442
    txt0="Result:"
    txt1="2 se/fl:" +str(int(fl))+"/"+str(int(fl_full))
    cv2.putText(image,txt0,(10,80),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),4)
    cv2.putText(image,txt1,(5,135),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),4)
    return image,fl_full
def draw_matches_on_images_lack_one(image, matched_pairs):#
    matches_line=[]
    fl=0

    for matches in matched_pairs:
        
        len(matched_pairs[0])
        flowerpot_coords = matches[0]
        flowerpot_x_min, flowerpot_y_min, flowerpot_x_max, flowerpot_y_max = flowerpot_coords[1:]
        center = ((flowerpot_x_min+flowerpot_x_max)//2,(flowerpot_y_min+flowerpot_y_max)//2)
        radius = min((flowerpot_x_max-flowerpot_x_min)//2,(flowerpot_y_max-flowerpot_y_min)//2)
        
        if image is not None:
           cv2.circle(image, center, radius, (255,0,0), thickness=14)
           cv2.rectangle(image, (flowerpot_x_min,flowerpot_y_min),(flowerpot_x_max,flowerpot_y_max),  (255,0,0),14)
         
        else:
            print("image is None")
        
       # se+=1
        plant_coords = matches[1]
        plant_x_min, plant_y_min, plant_x_max, plant_y_max = plant_coords[1:]
            
        fl+=1
    text_to_diaplay="1 plant/pot: "+str(fl)

    y =170#          
    cv2.putText(image,text_to_diaplay,(20,200),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),4)
    fl_one=str(fl)
    return image,fl_one

def draw_matches_on_images_empty(image, matched_pairs):#
    matches_line=[]
    fl=0
    se=0
    for pairs in matched_pairs:
        flowerpot_coords = pairs
        flowerpot_x_min, flowerpot_y_min, flowerpot_x_max, flowerpot_y_max = flowerpot_coords[1:]
        cv2.rectangle(image, (flowerpot_x_min,flowerpot_y_min),(flowerpot_x_max,flowerpot_y_max),(255,255,0), 14)

    values= len(matched_pairs)  
    
    txt="0 plant/pot: "+ str(values)
    fl_zero=str(values)
    cv2.putText(image,txt,(20,275),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),4)
    return image,fl_zero     
def compute_DIOU(box1, box2):
 
    eps=1e-7#
    b1_x1, b1_y1, b1_x2, b1_y2 = box1#
    b2_x1, b2_y1, b2_x2, b2_y2 = box2#

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)# 
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)#

  
    inter_x_min = max(b1_x1, b2_x1)#0
    inter_y_min = max(b1_y1, b2_y1)#1
    inter_x_max = min(b1_x2, b2_x2)#2
    inter_y_max = min(b1_y2, b2_y2)#3

    # 
    inter = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)#
    #
    w1,h1= b1_x2 - b1_x1, b1_y2 - b1_y1
    w2,h2= b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + eps#
    
    center_x1 = (b1_x1 + b1_x2) / 2
    center_y1 = (b1_y1 + b1_y2) / 2
    center_x2 = (b2_x1 + b2_x2) / 2
    center_y2 = (b2_y1 + b2_y2) / 2
    d = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2#为d**2  

 
    iou= inter / union
    cw=max(b1_x2, b2_x2)-min(b1_x1, b2_x1)#
    ch=max(b1_y2, b2_y2)-min(b1_y1, b2_y1)# 
    c2 = cw**2+ch**2+eps#
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    diou =iou-rho2 / c2
    return diou,c2

def distrbute(pot_idx_list, fl_label, se_label, pot_seedling_store):
    pair = [] # [(pot_label, seedling_label),]
    for pot_i in pot_idx_list:
        for seedling_i, _, _ in pot_seedling_store[pot_i]:
            pair.append((fl_label[pot_i], se_label[seedling_i]))
    return pair   

def single_images(pathss,imgnames,current_folder):
   
    result_list_info=[]
    cls=[]
    img_list=os.listdir(imgnames)
    se_label=[]
    fl_label=[]
    se=0
    fl=0
    full=0
    lackone=0
    lacktwo=0
    imglist_path=[os.path.join(imgnames,img) for img in img_list]
    for img_per in imglist_path:
        images_so=os.path.basename(img_per)
        imgg = cv2.imread(img_per)
        h,w,_=imgg.shape
        img=imgg.copy()
        se_label=[]
        fl_label=[]
        if img is None:
            print("Please check if the input image is correct, the image cannot be loaded.")
        else:
            la_path=pathss
            imagenames = images_so.split('.')[0]+'.txt'
            la= os.path.join(la_path,imagenames)
            if os.path.exists(la):
                labell = np.loadtxt(la)#   
                label =np.atleast_2d(labell) 
                for line in label:
                    x_min=int(w*(line[1]-line[3]*0.5))
                    y_min=int(h*(line[2]-line[4]*0.5))
                    x_max=int(w*(line[1]+line[3]*0.5))
                    y_max=int(h*(line[2]+line[4]*0.5))  
                    #print(line[0])                                  
                    if line[0]==1:
                        se_label.append([line[0],x_min,y_min,x_max,y_max])
                    elif line[0]==0:
                        fl_label.append([line[0],x_min,y_min,x_max,y_max]) 
        se+=len(se_label) 
        fl+=len(fl_label)                   
        full,lack_one,empty,pot_seedling_store=plantDIOU(se_label,fl_label)
        full_pair=distrbute(full,fl_label,se_label,pot_seedling_store)
        lack_one_pair=distrbute(lack_one,fl_label,se_label,pot_seedling_store)
        empty_label=[fl_label[pot_i] for pot_i in empty]
        images,fl_full=draw_matches_on_images_full( img, full_pair)
        images,fl_one=draw_matches_on_images_lack_one(images, lack_one_pair)  
        images,fl_zero=draw_matches_on_images_empty( images, empty_label)        
        result_image_path=current_folder+'/output/'+images_so
        
        cv2.imwrite(result_image_path, images)
        img_pth = result_image_path
        to= str(fl_full+fl_one+fl_zero)
        final_images_info={'path':img_pth,'stats':{'total':len(fl_label),'full':(fl_full),'lack_one':int(fl_one),'lack_two':int(fl_zero)}}
    #final_images_info = {'total': len(fl_label),'full': int(fl_full), 'lack_one':int(fl_one),'lack_two':int(fl_zero)}
                    
    return final_images_info
    
            
if __name__ == '__main__':
   labels_path= './runs/detect/exp/labels'
   img='./figure/test'
   current_folder = os.path.dirname(os.path.abspath(__file__))
   single_images(labels_path,img,current_folder)
  

