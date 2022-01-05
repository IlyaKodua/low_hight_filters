import cv2
import numpy as np
from utils import*

img_path = './PR.jpg'
img = cv2.imread(img_path)

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = np.expand_dims(img, axis=2)
save_img("etalon.png", img)
hight = get_filter_image(img, False)

save_img("hight.png", hight)

low = get_filter_image(img, True)


save_img("low.png", low)

thres = np.zeros_like(low)
thres[low > 100] = 255

save_img("thres.png", thres)
save_img("sum.png", low + hight)