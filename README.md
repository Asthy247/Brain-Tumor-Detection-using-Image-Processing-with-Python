# Brain-Tumor-Detection-using-Image-Processing-with-Python

# Introduction

This project aims to explore various image processing techniques to aid in the detection of brain tumors. 

We will utilize the Brain Tumor Detection dataset available on Kaggle, 

along with Python programming language and libraries like OpenCV, NumPy, and Scikit-image.

# Dataset

The Brain Tumor Detection dataset from Kaggle provides a rich source of medical images,

including both tumor-positive and tumor-negative cases. 

This dataset will be used to train and evaluate our image processing models.


**Dataset URL Link** : https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection

# Image Processing Techniques

**1. Edge Detection:**

**Sobel Filter**: Detects edges based on intensity gradients.

**Prewitt Filter:** Similar to Sobel but with a slightly different kernel.

**Laplacian Filter:** Detects edges based on the second derivative of intensity.


**2. Feature Extraction:**

**Gabor Filter:** Extracts features based on frequency and orientation.

**Hessian Matrix**: Provides information about the local curvature of the image intensity function.

# Data Visualization

# Converting Image from BRG to Gray (Visualizing Image Processing Techniques)

![image](https://github.com/user-attachments/assets/ba07081c-0cce-476b-8819-d900d5e6f2a7)

A code was created and ran in python for the dimensions of the grayscale image. 

The output (224, 224) indicates the dimensions of the grayscale image. It means the image has a height of 224 pixels and a width of 224 pixels.


# Image Visualization Prepped for Color Channel

![image](https://github.com/user-attachments/assets/3e1d65e3-f283-479e-a2d8-2a19bc02ad3d)

The output (224, 224, 3) indicates that the image o is a color image with three color channels (RGB). Each channel has a dimension of 224x224 pixels.

**Breakdown:**

**(224, 224):** Represents the height and width of the image in pixels.

**3:** Represents the number of color channels (Red, Green, Blue).

# Image Processing Techniques Using Filters

# visualization for Both The Original and Entropy Images Side-by-Side

![image](https://github.com/user-attachments/assets/2f1b120b-756c-48b6-aa2d-18a197e43bbd)



A code was created and ran that calculates the local entropy of an image, which can be a useful feature for image analysis tasks like texture classification or anomaly detection.

**Left Image:**

•	This image appears to be a visualization of the local entropy of the original image.

•	Entropy is a measure of disorder or randomness in a signal. In this case, it highlights areas of the image with higher variations in pixel intensity.

•	The brighter regions in the image indicate areas with higher entropy, suggesting more complex patterns or textures in those regions.


**Right Image:**

•	This image is likely the original brain image.

•	It shows the raw intensity values of the image pixels.

•	The darker regions might represent areas with lower tissue density or lower signal intensity.


# Gaussian Filter for Visualization for Both The Original and Filtered Versions Side-by-Side

A code was created for that applies a Gaussian filter to an image, which is a common technique for smoothing noise while preserving edges. 

The filtered image is then visualized alongside the original image for comparison.

![image](https://github.com/user-attachments/assets/9b01d865-4bc1-4acb-bce8-f8f83b997a11)


**Left Image:** The Gaussian-filtered image (gaussian_img), which will be a grayscale image with blurred details depending on the chosen sigma value.

**Right Image**: The original image (og).

Gaussian filtering is often used for noise reduction or pre-processing for other image analysis tasks.


# Sobel Filter for Visualization for Both The Original and Filtered Versions Side-by-Side

![image](https://github.com/user-attachments/assets/2a8987ca-6721-408b-9174-9c925dedf11e)

The output will consist of one or two images side-by-side:

**•	Left Image (if ax2.imshow(og) is uncommented):** The original image (og).


**•	Right Image:** The Sobel-filtered image (sobel_img), which will be a grayscale image where brighter regions represent stronger edges in the original image.


# Laplacian filter Used For Edge Detection or Image Sharpening

![image](https://github.com/user-attachments/assets/03317453-0532-4c5a-b122-6dc1c53a2de3)

The output consist of two images side-by-side:

**Left Image**: The Laplacian filtered image (laplace_img). Brighter regions might represent edges or areas with high intensity 

changes in the original image. Darker regions might indicate areas with smoother intensity variations. 


**Right Image**: The original image (og) for reference.

# Gabor Filtering for Image Orientation

![image](https://github.com/user-attachments/assets/470876b4-c5ab-49d4-843c-383a0ac36d6f)

The output consist of two images side-by-side:

**•	Left Image (if ax2.imshow(og) is uncommented)**: The original image (og).


**•	Right Image:** The Gabor-filtered image (gabor_img), which will be a grayscale image highlighting specific frequency components

in the original image based on the chosen filter parameters.


# Hessian Matrix of The Image at Multiple Scales

![image](https://github.com/user-attachments/assets/02768d32-39cf-4da1-bc9d-06998bc277aa)


# Prewitt Filter for Data Visualization on Both the Original and Edge-Detected Version


![image](https://github.com/user-attachments/assets/2216cbc3-39f2-4ff2-ac0e-0dc1ba10414e)


The output will consist of one or two images side-by-side:

•	**Left Image (if ax2.imshow(og) is uncommented)**: The original image (og).

•	**Right Image**: The Prewitt-filtered image (prewitt_img), which will be a grayscale image where brighter regions represent stronger 

edges in the horizontal or vertical directions in the original image.


# Results and Analysis

By applying these techniques to the brain tumor dataset, we can potentially identify regions of interest 

that may indicate the presence of a tumor. 

# Recommendations

**Experiment with Filter Parameters:** Adjust parameters like sigma, frequency, and orientation for the Gabor filter to tailor it to specific image characteristics.


**Combine Multiple Filters**: Combine the results of different filters to obtain more robust and informative feature representations.


**Consider Advanced Technique**s: Explore advanced techniques like adaptive thresholding, morphological operations, and machine learning algorithms for further image analysis and understanding.


**Evaluate Performance**: Use appropriate metrics like accuracy, sensitivity, specificity, and F1-score to evaluate the performance of different techniques.


**Consult with Medical Experts:** Collaborate with medical professionals to validate the results and ensure clinical relevance.

# Conclusion

This project has demonstrated the application of various image processing techniques to extract meaningful information from images. 

By effectively combining these techniques, we can gain valuable insights and automate tasks in various fields, 

including medical image analysis, computer vision, and pattern recognition.
