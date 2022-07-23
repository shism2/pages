import cv2
import os


in_dir = "thumb_add_here"
out_dir = "thumb_sized"

files = os.listdir(in_dir)

for f in files:
    img = cv2.imread(os.path.join(in_dir,f))
    h,w,c = img.shape

    target_width = 320
    target_height = int(h/w*target_width)

    img_new = cv2.resize(img,(target_width,target_height),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(out_dir,f),img_new)
