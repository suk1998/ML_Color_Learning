import numpy as np
import colour 
from sklearn.ensemble import RandomForestRegressor
# Library for creating 3D LUT
from colour.io import write_LUT_IridasCube
from colour import LUT3D
import os

# Functions for converting between sRGB and XYZ color spaces
def sRGB_to_XYZ(img):
    return colour.sRGB_to_XYZ(img)

def XYZ_to_sRGB(XYZ):
    return colour.XYZ_to_sRGB(XYZ)

# Define the paths for the original and transformed images
original_folder_path = './FHD_source'
transformed_folder_path = './resized_modiSCUNET_realFHD_data'

# Get all image file names from the folders
original_file_names = os.listdir(original_folder_path)
transformed_file_names = os.listdir(transformed_folder_path)

# Generate the full paths to the image files
original_image_paths = [os.path.join(original_folder_path, file) for file in original_file_names if file.lower().endswith(('.png','.jpg'))]
transformed_image_paths = [os.path.join(transformed_folder_path, tfile) for tfile in transformed_file_names if tfile.lower().endswith(('.png','.jpg'))]

# Lists to store the image data in XYZ color space
all_original_XYZ = []
all_transformed_XYZ = []

# Convert images to XYZ and store them
for original_path, transformed_path in zip(original_image_paths, transformed_image_paths):
    original_img = colour.read_image(original_path)
    original_XYZ = sRGB_to_XYZ(original_img)
    all_original_XYZ.append(original_XYZ.reshape(-1,3))

    transformed_img = colour.read_image(transformed_path)
    transformed_XYZ = sRGB_to_XYZ(transformed_img)
    all_transformed_XYZ.append(transformed_XYZ.reshape(-1,3))

# Flatten the lists of XYZ data for training
all_original_XYZ_flat = np.vstack(all_original_XYZ)
all_transformed_XYZ_flat = np.vstack(all_transformed_XYZ)

# Initialize and train the RandomForestRegressor model
rf_model = RandomForestRegressor(n_estimators=150, random_state=42)
rf_model.fit(all_transformed_XYZ_flat, all_original_XYZ_flat)

# Creating a 3D LUT and transforming it
lut_RGB = LUT3D.linear_table(127)  # Generate a linear 3D LUT
lut_XYZ = colour.sRGB_to_XYZ(lut_RGB)  # Convert the LUT to XYZ color space
lut_XYZ_reshaped = lut_XYZ.reshape(-1, 3)  # Reshape for prediction

# Use the trained model for prediction and reshape back
transformed_lut_XYZ = rf_model.predict(lut_XYZ_reshaped).reshape(lut_XYZ.shape)

# Convert the predicted XYZ colors back to sRGB and clip values
transformed_lut_sRGB = XYZ_to_sRGB(transformed_lut_XYZ)
transformed_lut_sRGB = np.clip(transformed_lut_sRGB, 0, 1)

# Save the transformed LUT
LUT = LUT3D(transformed_lut_sRGB, 'My LUT', comments=['BT709_to_BT709.', 'grid=127'])
write_LUT_IridasCube(LUT, 'realFHDdata_color_transformation_randomforests_150_modiscunet_x2.cube')

print("transformed_lut_sRGB's shape: ", transformed_lut_sRGB.shape)
