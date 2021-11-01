import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr
from PIL import Image 
import PIL.ImageOps
x,y=fetch_openml("mnist_784",version=1,return_X_y=True)
alphabets=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses=len(alphabets)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=7500,test_size=2500,random_state=40)
x_train=x_train/255.0
x_test=x_test/255.0
clf=lr(solver="saga",multi_class="multinomial").fit(x_train,y_train)
def getPrediction(image):
    im_pil=Image.open(image)
    im_bw=im_pil.convert("L")
    im_bw_resized=im_bw.resize((22,30),Image.ANTIALIAS)
    pixel_filter=20
    min_pixel=np.percentile(im_bw_resized,pixel_filter)
    im_bw_inverted=np.clip(im_bw_resized-min_pixel,0,255)
    max_pixel=np.max(im_bw_resized)
    im_bw_inverted=np.asarray(im_bw_inverted)/max_pixel
    test_sample=np.array(im_bw_inverted).reshape(1,660)
    pred=clf.predict(test_sample)
    return pred[0]