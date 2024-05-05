import numpy as np
import colour 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
# Library for creating 3D LUT
from colour.io import write_LUT_IridasCube
from colour import LUT3D
import matplotlib.pyplot as plt
import os

# XYZ 독립적인 공간으로 변형해서 하는 경우
def sRGB_to_XYZ(img):
    img_XYZ = colour.sRGB_to_XYZ(img)
    return img_XYZ

def XYZ_to_sRGB(XYZ):
    img_sRGB = colour.XYZ_to_sRGB(XYZ)
    return img_sRGB

original_folder_path = './resized_DIV2K'

transformed_folder_path = './LGU_scunet_x1'

original_file_names = os.listdir(original_folder_path)

transformed_file_names = os.listdir(transformed_folder_path)

original_image_paths = [os.path.join(original_folder_path, file) for file in original_file_names if file.lower().endswith(('.png','.jpg'))]

transformed_image_paths = [os.path.join(transformed_folder_path, tfile) for tfile in transformed_file_names if tfile.lower().endswith(('.png','.jpg'))]

all_original_XYZ = []
all_transformed_XYZ = []

for original_path, transformed_path in zip(original_image_paths, transformed_image_paths):
    original_img = colour.read_image(original_path)
    original_XYZ = sRGB_to_XYZ(original_img)
    all_original_XYZ.append(original_XYZ.reshape(-1,3))

    transformed_img = colour.read_image(transformed_path)
    transformed_XYZ = sRGB_to_XYZ(transformed_img)
    all_transformed_XYZ.append(transformed_XYZ.reshape(-1,3))

# # 모든 이미지의 데이터를 결합
all_original_XYZ_flat = np.vstack(all_original_XYZ)
all_transformed_XYZ_flat = np.vstack(all_transformed_XYZ)

degree = 2

poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

poly_model.fit(all_transformed_XYZ_flat, all_original_XYZ_flat)

# creating 3D LUT
lut_RGB = LUT3D.linear_table(127)

lut_XYZ = colour.sRGB_to_XYZ(lut_RGB)
# RGB 색상값의 1차원 배열로 변환
lut_XYZ_reshaped = lut_XYZ.reshape(-1, 3)

# 변환된 XYZ 값을 저장할 배열 초기화
transformed_lut_XYZ_reshaped = np.zeros_like(lut_XYZ_reshaped)

transformed_lut_XYZ = poly_model.predict(lut_XYZ_reshaped).reshape(lut_XYZ.shape)

transformed_lut_sRGB = XYZ_to_sRGB(transformed_lut_XYZ)

transformed_lut_sRGB = np.clip(transformed_lut_sRGB, 0, 1)

print("transformed_lut_sRGB's shape: ", transformed_lut_sRGB.shape)

LUT = LUT3D(
    transformed_lut_sRGB,
    'My LUT',
#     domain,

    comments=['BT709_to_BT709.', 'grid=127'])

write_LUT_IridasCube(LUT, 'color_transformation_poly_LGUscunet_x1.cube')