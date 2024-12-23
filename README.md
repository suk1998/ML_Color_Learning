Category: FHD to 4K Upscaling / Color Leanring through ML to reproduce FHD color
Technologies Used: Random Forest Regressor(trees=100), AI Super Resolution: SCUNET
Date: Februrary 2024

It has always been mentioned that color shifting occurs when passing through AI upscaling models. SBS requested to minimize such color shifts and ensure that, 
even after upscaling, the original colors are accurately reproduced.
I found myself wondering whether AI upscaling models not only enhance image quality but also improve color. Upon comparing color points in the color space, 
I discovered that AI upscaling models don't enhance colors; instead, they irregularly shift color points left and right. Mapping these color points mathematically posed difficulties, 
so instead, I thought of a method: training colors through machine learning. I first converted color values from the range of 0 to 255 to the range of 0 to 1 (converting int8 to float64) and 
then transformed device-dependent RGB values into device-independent XYZ values. Subsequently, I trained a Random Forest Regressor with 800 samples of both DIV2K original image data and 
images processed through the AI upscaling model. I used the resulting weights as a lookup table to apply real-time color correction to the AI upscaled video affected by color shifts. 
However, the learning through machine learning didn't perfectly reproduce the original colors; instead, it reproduced intermediate colors between the color-shifted upscaled model and the original. 
I realized that this aspect warrants further research.
