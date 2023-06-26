from matplotlib.image import imsave
import os


for filepict in teethlist:
  ## Plotting - RESULT Example with CCA_Analysis
  entity = " ".join(filepict.split("_"))
  filename = f"{entity}.bmp"
  project_path = f"drive/MyDrive/Data_baru_raihan/{filepict}_100rsgm/"
  instance_name = project_path.split("/")[3]

  img = cv2.imread("drive/MyDrive/Data_baru_raihan/111_TeethList/" + filename)#original img 107.png 
  # img = cv2.imread("/content/bl_cej_PX20170317_214518_0000_ED00030F.bmp") #original img 107.png 

  # load image (mask was saved by matplotlib.pyplot) 
  teeth_pics = os.listdir(project_path)
  teeth_pics.sort()

  bl = cv2.imread(f"{project_path}{teeth_pics[0]}".format(filename.split(".")[0]))
  bl_copy = cv2.imread(f"{project_path}{teeth_pics[0]}".format(filename.split(".")[0]))
  bottom_cej = cv2.imread(f"{project_path}{teeth_pics[1]}".format(filename.split(".")[0]))
  bottom_teeth = cv2.imread(f"{project_path}{teeth_pics[2]}".format(filename.split(".")[0]))
  top_cej = cv2.imread(f"{project_path}{teeth_pics[3]}".format(filename.split(".")[0]))
  top_teeth = cv2.imread(f"{project_path}{teeth_pics[4]}".format(filename.split(".")[0]))
  black_img = np.zeros(img.shape, dtype="uint8")

  top_teeth = cv2.resize(top_teeth, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
  bottom_teeth = cv2.resize(bottom_teeth, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
  top_cej = cv2.resize(top_cej, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
  bottom_cej = cv2.resize(bottom_cej, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
  bl = cv2.resize(bl, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

  cca_result, teeth_count, stage_labels_dict = CCA_Analysis(img, bl, bl_copy, top_cej, top_teeth, bottom_cej, bottom_teeth, 4, 2)
  stage_labels_dict = sorted(stage_labels_dict.items(), key=lambda x: x[1], reverse=True)
  cv2_imshow(cca_result)
  stage_labels_dict.sort()

  print("Stage counts:")
  for item in stage_labels_dict:
    print("-", item[0]+":",item[1])
  print("Teeth count:", teeth_count)
  loc_file = f"{project_path}" + f"result_{instance_name}.bmp"
  imsave(loc_file, np.uint8(cca_result))