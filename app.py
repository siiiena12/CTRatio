"""Streamlit web app for Chest segmentation"""

from collections import namedtuple

import os
import cv2
import pydicom
import pandas as pd
import numpy as np
from numpyencoder import NumpyEncoder
import json
import streamlit as st
import albumentations as A
import torch
from torch import nn
from torch.utils import model_zoo
import unet as Unet

import tensorflow as tf
import keras.backend as keras
from keras.models import *
from keras.layers import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


st.set_option("deprecation.showfileUploaderEncoding", False)

img_size = 512
aug = A.Compose([A.Resize(img_size, img_size, interpolation=1, p=1)], p=1)

model = namedtuple("model", ["url", "model"])
models = {
    "resnet34": model(
        url="https://github.com/alimbekovKZ/lungs_segmentation/releases/download/1.0.0/resnet34.pth",
        model=Unet.Resnet(seg_classes=2, backbone_arch="resnet34"),
    ),
    "densenet121": model(
        url="https://github.com/alimbekovKZ/lungs_segmentation/releases/download/1.0.0/densenet121.pth",
        model=Unet.DensenetUnet(seg_classes=2, backbone_arch="densenet121"),
    ),
}

def create_model(model_name: str) -> nn.Module:
    model = models[model_name].model
    state_dict = model_zoo.load_url(
        models[model_name].url, progress=True, map_location="cpu"
    )
    model.load_state_dict(state_dict)
    return model


@st.cache(allow_output_mutation=True)
def cached_model():
    model = create_model("resnet34")
    device = torch.device("cpu")
    model = model.to(device)
    return model


model = cached_model()

st.title("Cardiomegaly 자동 측정 알고리즘")

# uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
file = st.sidebar.file_uploader(
    "Upload your image (dicom or png)", ["png","dcm"]
)


def lung_inference(model, image, thresh=0.2):
    model.eval()
    image = (image - image.min()) / (image.max() - image.min())
    augs = aug(image=image)
    image = augs["image"].transpose((2, 0, 1))
    im = augs["image"]
    image = np.expand_dims(image, axis=0)
    image = torch.tensor(image)

    mask = torch.nn.Sigmoid()(model(image.float()))
    mask = mask[0, :, :, :].cpu().detach().numpy()
    mask = (mask > thresh).astype("uint8")
    torch.cuda.empty_cache() #캐시데이터삭제
    return im, mask

def img_with_masks(img, masks, alpha, return_colors=False):
    """
    returns image with masks,
    img - numpy array of image
    masks - list of masks. Maximum 6 masks. only 0 and 1 allowed
    alpha - int transparency [0:1]
    return_colors returns list of names of colors of each mask
    """
    # colors = [
    #     [255, 0, 0],
    #     [0, 255, 0],
    #     [0, 0, 255],
    #     [255, 255, 0],
    #     [0, 255, 255],
    #     [102, 51, 0],
    # ]
    colors = [
        [255, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 255, 255],
        [0, 0, 255],       
        [255, 255, 0],
        [102, 51, 0],
        
    ]
    color_names = ["Red", "Red", "Green","Light","BLue", "Yello", "Brown"]
    img = img - img.min()
    img = img / (img.max() - img.min())
    img *= 255
    img = img.astype(np.uint8)

    c = 0
    for mask in masks:
        mask = np.dstack((mask, mask, mask)) * np.array(colors[c])
        mask = mask.astype(np.uint8)
        img = cv2.addWeighted(mask, alpha, img, 1, 0.0)
        c = c + 1
    if return_colors is False:
        return img
    else:
        return img, color_names[0 : len(masks)]
      
def CXR_findTD(maskL,maskR):
    maskL = maskL.astype(np.uint8)*255
    maskR = maskR.astype(np.uint8)*255

    # 가장자리만 검출하는 방법을 이용하여 왼쪽폐의 최소x값, 오른쪽폐 최대x값 구하기
    cannyL = cv2.Canny(maskL,127,255)
    yCannyL,xCannyL = np.where(cannyL>0)
    cannyR = cv2.Canny(maskR,127,255)
    yCannyR,xCannyR = np.where(cannyR>0)
    
    return {
        "left" : [np.min(xCannyL),yCannyL[np.argmin(xCannyL)]],
        "right" : [np.max(xCannyR),yCannyR[np.argmax(xCannyR)]],
        "length" : np.max(xCannyR)-np.min(xCannyL),
        "center" : (np.max(xCannyR)+np.min(xCannyL))//2,
        "yMax" : np.max([np.max(yCannyL),np.max(yCannyR)])
    }       

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def image_to_train(img):
    npy = img / 255
    npy = np.reshape(npy, npy.shape + (1,))
    npy = np.reshape(npy,(1,) + npy.shape)
    return npy

def train_to_image(npy):
    img = (npy[0,:, :, 0]).astype(np.uint8)
    return img

def calc_len(msk):
    # 노이즈 제거
    contours,_ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    xDataset = pd.DataFrame(columns=['xMin','yMin','xMax','yMax','xLen'])
    for contour in contours:
        contour = contour.squeeze(1)
        msk[contour.T[1],contour.T[0]]=1

        x,y,w,h = cv2.boundingRect(contour)
        xDataset = xDataset.append({
                "xMin": x,
                "yMin": y,
                "xMax": x+w,
                "yMax": y+h,
                "xLen" : w,
            }, ignore_index=True)
    
    result_idx = np.argmax(xDataset.xLen)
    nois1 = xDataset.loc[:result_idx-1]
    nois2 = xDataset.loc[result_idx+1:]

    if len(nois1)>0:
        for i in nois1.index:
            cv2.rectangle(msk, (nois1.xMin[i], nois1.yMin[i]), (nois1.xMax[i], nois1.yMax[i]), (0,0,0), -1)
    if len(nois2)>0:
        for i in nois2.index:
            cv2.rectangle(msk, (nois2.xMin[i], nois2.yMin[i]), (nois2.xMax[i], nois2.yMax[i]), (0,0,0), -1)

    return msk

def cardiac_inference(himage):
    heart = cv2.resize(himage, (512, 512))
    cardiac_model = load_model('./cardiac_segmentation/[512png_contrast]unet_cardiac_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef': dice_coef})
    # cardiac_model.summary()
    segm_ret = cardiac_model.predict(image_to_train(heart), verbose=0)
    msk = train_to_image(segm_ret)
    
    return calc_len(msk)

def CXR_findMD(mask):
    mask = mask.astype(np.uint8)*255
    # 가장자리만 검출하는 방법을 이용하여 심장 최소x값, 심장 최대x값 구하기
    canny = cv2.Canny(mask,127,255)
    yCanny,xCanny = np.where(canny>0)
    
    return {
        "left" : [np.min(xCanny),yCanny[np.argmin(xCanny)]],
        "right" : [np.max(xCanny),yCanny[np.argmax(xCanny)]],
        "length" : np.max(xCanny)-np.min(xCanny)
    } 

# 이미지를 더 선명하게 Contrast 기법적용
def img_Contrast(img):
    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    return final


if file is not None:
    base_file, fileext = os.path.splitext(os.path.basename(file.name))

    oimage = ''
    if '.dcm' in fileext:
        oimage = pydicom.dcmread(file).pixel_array
        if oimage.dtype != 'uint8':
            oimage = cv2.normalize(oimage, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        oimage = np.repeat(oimage[:,:,np.newaxis],3,-1) #image 3채널 추가

    else:
        oimage = cv2.imdecode(np.fromstring(file.read(), np.uint8), 1)

    oimage = cv2.cvtColor(oimage, cv2.COLOR_BGR2RGB)
    st.image(oimage, caption="Original Image", use_column_width=True)
    st.write("")
    st.write("Result Loading...") 

    ocimage = img_Contrast(oimage)    
    himage = cv2.cvtColor(ocimage, cv2.COLOR_RGB2GRAY)

    image, mask = lung_inference(model, ocimage, 0.2)
    maskC = cardiac_inference(himage)

    torch.cuda.empty_cache() #캐시데이터삭제

    mask[0] = calc_len(mask[0].astype(np.uint8))
    mask[1] = calc_len(mask[1].astype(np.uint8))

    lungMask = CXR_findTD(mask[0],mask[1])
    cardiacMask = CXR_findMD(maskC)
    
    merge2img = img_with_masks(image, [mask[0], mask[1], maskC], alpha=0.3)
    # st.caching.clear_cache()

    # 선 그리기 및 글자표기
    testImg = merge2img.copy()
    color = (0,0,255)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    cv2.line(testImg,(cardiacMask['left'][0],cardiacMask['left'][1]),(lungMask['center'],cardiacMask['left'][1]),color, 2)
    cv2.line(testImg,(lungMask['center'],cardiacMask['right'][1]),(cardiacMask['right'][0],cardiacMask['right'][1]),color, 2)
    cv2.line(testImg,(lungMask['left'][0],lungMask['yMax']),(lungMask['right'][0],lungMask['yMax']),(255, 255, 0), 2)

    MLD = lungMask['center']-cardiacMask['left'][0]
    MRD = cardiacMask['right'][0]-lungMask['center']
    TD = lungMask['length']
    cv2.putText(testImg,f'MLD : {MLD}',(cardiacMask['left'][0],cardiacMask['left'][1]-20),font,0.5,color,1,cv2.LINE_AA)
    cv2.putText(testImg,f'MRD : {MRD}',(lungMask['center']+10,cardiacMask['right'][1]-20),font,0.5,color,1,cv2.LINE_AA)
    cv2.putText(testImg,f'TD : {TD}',(TD//2,lungMask['yMax']-20),font,0.7,(255, 255, 0),1,cv2.LINE_AA)
    
    st.image(testImg, caption="Image + mask", use_column_width=True)
    
    # 비율 나타내기
    ctratio = ((MLD+MRD)/TD)*100
    st.write(f'CTRatio : {round(ctratio,2)}%')

    if ctratio < 50.0:
        st.write("정상입니다.")
    else:
        st.write("비정상입니다.")

    # json으로 내보내기
    dict = {
        "filename" : file.name,
        "lung" : {
            "x1y1" : lungMask['left'],
            "x2y2" : lungMask['right']
        },
        "heart" : {
            "x1y1" : cardiacMask['left'],
            "x2y2" : cardiacMask['right']
        },
        "center" : {
            "x1y1" : [lungMask['center'],0],
            "x2y2" : [lungMask['center'],oimage.shape[0]]
        }
    }
    st.download_button(
        label="Download JSON", 
        data=json.dumps(dict, cls= NumpyEncoder,ensure_ascii=False, indent=4), 
        file_name='cardiomegaly.json', 
        mime='application/json')
    
