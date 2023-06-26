import cv2
import numpy as np
from imutils import perspective
from scipy.spatial import distance as dist
from google.colab.patches import cv2_imshow

def midpoint(ptA, ptB):
	  return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def draw_outline(black_img, masking_img):
    mask = cv2.resize(masking_img, (black_img.shape[1],black_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask_thresh = cv2.threshold(mask_gray, 150, 255, cv2.THRESH_BINARY)
    contours_mask, hierarchy = cv2.findContours(image=mask_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image=black_img, contours=contours_mask, contourIdx=-1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    return black_img

def stage_conversion(ptg_stage):
    if ptg_stage < 18:
        stage = "N"
    elif ptg_stage >= 18 and ptg_stage <= 36:
        stage = "S2"
    elif ptg_stage > 36:
        stage = "S3"
    else:
        stage = "Error"
    return stage

def find_joints(res_img, image, contra_image, loc, root_coord):
    # Find joints
    joints = cv2.bitwise_and(image, contra_image)

    # Find centroid of the joints
    joints_gray = cv2.cvtColor(joints, cv2.COLOR_BGR2GRAY)
    ret, joints_thresh = cv2.threshold(joints_gray, 150, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(joints_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    joints_coord = []
    for c in cnts:
        # Find centroid and draw center point
        M = cv2.moments(c)
        try:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # print("(cx, cy)", (cx, cy))
            joints_coord.append((cx, cy))

        except ZeroDivisionError:
            continue
    
    if len(joints_coord) == 3:
        if loc == "top":
            joints_coord = joints_coord[1:]
        else:
            joints_coord = joints_coord[:-1]
        
        cej_bl_dist = dist.euclidean(joints_coord[0], joints_coord[1])
        cej_root_dist = dist.euclidean(joints_coord[0], root_coord)
        # print("cej_bl_dist", cej_bl_dist)
        # print("cej_root_dist", cej_root_dist)
        ptg_stage = float((cej_bl_dist/cej_root_dist)*100)
        # print("ptg_stage", ptg_stage)
        stage_label = stage_conversion(ptg_stage)
        ptg_label = "{:.2f}%".format(ptg_stage)

    elif len(joints_coord) == 2:
        cej_bl_dist = dist.euclidean(joints_coord[0], joints_coord[1])
        cej_root_dist = dist.euclidean(joints_coord[0], root_coord)
        # print("cej_bl_dist", cej_bl_dist)
        # print("cej_root_dist", cej_root_dist)
        ptg_stage = float((cej_bl_dist/cej_root_dist)*100)
        # print("ptg_stage", ptg_stage)
        stage_label = stage_conversion(ptg_stage)
        ptg_label = "{:.2f}%".format(ptg_stage)
    else:
        stage_label = "N"
        ptg_label = ""
  
    # print("joints_coord", joints_coord)
    for coord in joints_coord:
        cv2.circle(res_img, coord, 5, (255,0,0), -1)

    if loc == "top":
        cv2.putText(res_img, stage_label, (root_coord[0], root_coord[1]-35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(res_img, ptg_label, (root_coord[0], root_coord[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    else:
        cv2.putText(res_img, stage_label, (root_coord[0], root_coord[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(res_img, ptg_label, (root_coord[0], root_coord[1]+45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    # cv2_imshow(joints)
    return res_img, stage_label
 
def CCA_Analysis(orig_image, bl, bl_copy, top_cej, top_teeth, bottom_cej, bottom_teeth, erode_iteration, open_iteration):
    kernel1 = (np.ones((5,5), dtype=np.float32))
    kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1, 9,-1], 
                                  [-1,-1,-1]])
    stage_labels = []
    res_img = draw_outline(orig_image, bl) # add BL

    black_img = np.zeros(orig_image.shape, dtype="uint8")
    black_img2 = np.zeros(orig_image.shape, dtype="uint8")
    black_img = draw_outline(black_img, bl) # add BL
    black_img = draw_outline(black_img, top_cej) # add CEJ
    res_img = draw_outline(orig_image, top_cej) # add CEJ
    image = top_teeth 
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel1,iterations=open_iteration )
    image = cv2.filter2D(image, -1, kernel_sharpening)
    image = cv2.erode(image,kernel1,iterations =erode_iteration)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    labels = cv2.connectedComponents(thresh, connectivity=8)[1]
    a = np.unique(labels)
    count2 = 0
    for label in a:
        if label == 0:
            continue

        # Create a mask
        mask = np.zeros(thresh.shape, dtype="uint8")
        mask[labels == label] = 255
        # Find contours and determine contour area
        cnts, hieararch = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_copy = black_img.copy()
        cv2.drawContours(image=image_copy, contours=cnts, contourIdx=-1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cnts = cnts[0]
        c_area = cv2.contourArea(cnts)
        
        # threshhold for tooth count
        if c_area>2000 and c_area<20000:
            count2+=1
        
            (x,y),radius = cv2.minEnclosingCircle(cnts)
            rect = cv2.minAreaRect(cnts)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="int")    
            box = perspective.order_points(box) 
            # cv2.drawContours(image2,[box.astype("int")],0,color,2)
            (tl,tr,br,bl) = box
            
            (tltrX,tltrY) = midpoint(tl,tr)
            (blbrX,blbrY) = midpoint(bl,br)
          # compute the midpoint between the top-left and top-right points,
          # followed by the midpoint between the top-righ and bottom-right
            (tlblX,tlblY) = midpoint(tl,bl)
            (trbrX,trbrY) = midpoint(tr,br)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            pixelsPerMetric = 1
            dimA = dA * pixelsPerMetric
            dimB = dB * pixelsPerMetric
            if dimA > dimB:
                black_img2 = np.zeros(orig_image.shape, dtype="uint8")
                cv2.line(black_img2, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 255, 255), 5)
                cv2.line(res_img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 255, 255), 2)
                if int(tltrY) > int(blbrY):
                    cv2.circle(res_img, (int(tltrX), int(tltrY)), 5, (0, 255, 0), -1) # B, G, R
                    cv2.circle(res_img, (int(blbrX), int(blbrY)), 5, (0, 0, 255), -1) # B, G, R
                    root_coord = (int(blbrX), int(blbrY))
                else:
                    cv2.circle(res_img, (int(tltrX), int(tltrY)), 5, (0, 0, 255), -1) # B, G, R
                    cv2.circle(res_img, (int(blbrX), int(blbrY)), 5, (0, 255, 0), -1) # B, G, R
                    root_coord = (int(tltrX), int(tltrY))
                # print("tltrX, tltrY", (int(tltrX), int(tltrY)))
                # print("blbrX, blbrY", (int(blbrX), int(blbrY)))
            else:
                black_img2 = np.zeros(orig_image.shape, dtype="uint8")
                cv2.line(black_img2, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 255, 255), 5)
                cv2.line(res_img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 255, 255), 2)
                if int(tlblY) > int(trbrY):
                    cv2.circle(res_img, (int(tlblX), int(tlblY)), 5, (0, 255, 0), -1) # B, G, R
                    cv2.circle(res_img, (int(trbrX), int(trbrY)), 5, (0, 0, 255), -1) # B, G, R
                    root_coord = (int(trbrX), int(trbrY))
                else:
                    cv2.circle(res_img, (int(tlblX), int(tlblY)), 5, (0, 0, 255), -1) # B, G, R
                    cv2.circle(res_img, (int(trbrX), int(trbrY)), 5, (0, 255, 0), -1) # B, G, R
                    root_coord = (int(tlblX), int(tlblY))
                # print("tlblX, tlblY", (int(tlblX), int(tlblY)))
                # print("trbrX, trbrY", (int(trbrX), int(trbrY)))
            
            cv2_imshow(image_copy)
            image, stage_label = find_joints(res_img, black_img, black_img2, "top", root_coord)
            stage_labels.append(stage_label)

    black_img_bottom = np.zeros(orig_image.shape, dtype="uint8")
    black_img2_bottom = np.zeros(orig_image.shape, dtype="uint8")
    black_img_bottom = draw_outline(black_img_bottom, bl_copy) # add BL
    black_img_bottom = draw_outline(black_img_bottom, bottom_cej) # add CEJ
    res_img = draw_outline(orig_image, bottom_cej) # add CEJ
    image_bottom = bottom_teeth 
    image_bottom = cv2.morphologyEx(image_bottom, cv2.MORPH_OPEN, kernel1,iterations=open_iteration )
    image_bottom = cv2.filter2D(image_bottom, -1, kernel_sharpening)
    image_bottom = cv2.erode(image_bottom,kernel1,iterations =erode_iteration)
    image_bottom = cv2.cvtColor(image_bottom, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(image_bottom, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    labels = cv2.connectedComponents(thresh,connectivity=8)[1]
    a = np.unique(labels)
    count2_bottom = 0
    for label in a:
        if label == 0:
            continue

        # Create a mask
        mask = np.zeros(thresh.shape, dtype="uint8")
        mask[labels == label] = 255
        # Find contours and determine contour area
        cnts, hieararch = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        c_area = cv2.contourArea(cnts)
        
        # threshhold for tooth count
        if c_area>500 and c_area<20000:
            count2_bottom+=1
        
            (x,y),radius = cv2.minEnclosingCircle(cnts)
            rect = cv2.minAreaRect(cnts)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="int")    
            box = perspective.order_points(box) 
            # cv2.drawContours(image2,[box.astype("int")],0,color,2)
            (tl,tr,br,bl) = box
            
            (tltrX,tltrY) = midpoint(tl,tr)
            (blbrX,blbrY) = midpoint(bl,br)
          # compute the midpoint between the top-left and top-right points,
          # followed by the midpoint between the top-righ and bottom-right
            (tlblX,tlblY) = midpoint(tl,bl)
            (trbrX,trbrY) = midpoint(tr,br)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            pixelsPerMetric = 1
            dimA = dA * pixelsPerMetric
            dimB = dB * pixelsPerMetric
            if dimA > dimB:
                black_img2_bottom = np.zeros(orig_image.shape, dtype="uint8")
                cv2.line(black_img2_bottom, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 255, 255), 5)
                cv2.line(res_img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 255, 255), 2)
                if int(tltrY) > int(blbrY):
                    cv2.circle(res_img, (int(tltrX), int(tltrY)), 5, (0, 0, 255), -1) # B, G, R
                    cv2.circle(res_img, (int(blbrX), int(blbrY)), 5, (0, 255, 0), -1) # B, G, R
                    root_coord = (int(tltrX), int(tltrY))
                else:
                    cv2.circle(res_img, (int(tltrX), int(tltrY)), 5, (0, 255, 0), -1) # B, G, R
                    cv2.circle(res_img, (int(blbrX), int(blbrY)), 5, (0, 0, 255), -1) # B, G, R
                    root_coord = (int(blbrX), int(blbrY))
                # print("tltrX, tltrY", (int(tltrX), int(tltrY)))
                # print("blbrX, blbrY", (int(blbrX), int(blbrY)))
            else:
                black_img2_bottom = np.zeros(orig_image.shape, dtype="uint8")
                cv2.line(black_img2_bottom, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 255, 255), 5)
                cv2.line(res_img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 255, 255), 2)
                if int(tlblY) > int(trbrY):
                    cv2.circle(res_img, (int(tlblX), int(tlblY)), 5, (0, 0, 255), -1) # B, G, R
                    cv2.circle(res_img, (int(trbrX), int(trbrY)), 5, (0, 255, 0), -1) # B, G, R
                    root_coord = (int(tlblX), int(tlblY))
                else:
                    cv2.circle(res_img, (int(tlblX), int(tlblY)), 5, (0, 255, 0), -1) # B, G, R
                    cv2.circle(res_img, (int(trbrX), int(trbrY)), 5, (0, 0, 255), -1) # B, G, R
                    root_coord = (int(trbrX), int(trbrY))
                # print("tlblX, tlblY", (int(tlblX), int(tlblY)))
                # print("trbrX, trbrY", (int(trbrX), int(trbrY)))
            
            # cv2_imshow(cnts)
            image, stage_label = find_joints(res_img, black_img_bottom, black_img2_bottom, "bottom", root_coord)
            stage_labels.append(stage_label)

    teeth_count = count2 + count2_bottom
    stage_labels_dict = {}
    for item in stage_labels:
        stage_labels_dict[item] = stage_labels_dict.get(item, 0) + 1
    
    return res_img, teeth_count, stage_labels_dict