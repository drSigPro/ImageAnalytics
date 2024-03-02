#%% Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit app
st.divider()
st.title("Edge Detection Demo \t")
st.divider()

# Upload an audio file
uploaded_file = st.file_uploader("Upload an image")

# Kernel Size
GaussSize = st.sidebar.slider("Select Gaussian Kernel Size", min_value=3, max_value=51, value=5, step=2)
GradientSize = st.sidebar.slider("Select Gradient Kernel Size", min_value=3, max_value=51, value=3, step = 2)
lowThreshold = st.sidebar.slider("Lower Thresold", min_value=10, max_value=255, value=30)
highThreshold = st.sidebar.slider("Higher Thresold", min_value=10, max_value=255, value=100)

# Main Streamlit app
def main():
    
    if uploaded_file:

        if GaussSize or GradientSize or lowThreshold or highThreshold:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            
            Img=cv2.imdecode(file_bytes, 1)
            
            if (len(np.shape(Img))==3):
                Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
           
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(Img, (GaussSize, GaussSize), 0)

            # Compute gradients
            gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=GradientSize)
            gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=GradientSize)

            # Compute gradient magnitude and direction
            gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
            gradient_direction = cv2.phase(gradient_x, gradient_y, angleInDegrees=True)

            # Angle quantization
            angle_quantized = np.zeros_like(gradient_direction, dtype=np.uint8)
            angle_quantized[np.where((gradient_direction >= -22.5) & (gradient_direction < 22.5))] = 0
            angle_quantized[np.where((gradient_direction >= 22.5) & (gradient_direction < 67.5))] = 45
            angle_quantized[np.where((gradient_direction >= 67.5) & (gradient_direction < 112.5))] = 90
            angle_quantized[np.where((gradient_direction >= 112.5) & (gradient_direction < 157.5))] = 135
            angle_quantized[np.where((gradient_direction >= 157.5) | (gradient_direction < -157.5))] = 0

            # Perform non-maximum suppression
            gradient_magnitude_suppressed = cv2.morphologyEx(gradient_magnitude, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

            # Perform edge tracing by hysteresis
            edges = cv2.Canny(blurred, threshold1=lowThreshold, threshold2=highThreshold)  # You can adjust the thresholds as needed          
            
            # Display the results
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original Image
            axs[0, 0].imshow(Img, cmap='gray')
            axs[0, 0].set_title('Original Image')
            axs[0, 0].axis('off')
            
            # Gaussian Blur
            axs[0, 1].imshow(blurred, cmap='gray')
            axs[0, 1].set_title('Gaussian Blur')
            axs[0, 1].axis('off')
            
            # Gradient Magnitude
            axs[0, 2].imshow(gradient_magnitude, cmap='gray')
            axs[0, 2].set_title('Gradient Magnitude')
            axs[0, 2].axis('off')
            
            # Gradient Direction
            axs[1, 0].imshow(gradient_direction, cmap='gray')
            axs[1, 0].set_title('Gradient Direction')
            axs[1, 0].axis('off')
            
            # Non-maximum Suppression
            axs[1, 1].imshow(angle_quantized, cmap='gray')
            axs[1, 1].set_title('Quantized Gradient Directi')
            axs[1, 1].axis('off')
            
            # Final Edges (Canny)
            axs[1, 2].imshow(gradient_magnitude_suppressed, cmap='gray')
            axs[1, 2].set_title('Non-maximum Suppression')
            axs[1, 2].axis('off') 
            
            st.pyplot(fig)
            
            fig = plt.figure(figsize=(1, 1))
            plt.imshow(edges,cmap='gray')     
            plt.axis('off')  
            plt.title('Final Edges (Canny)')
            st.pyplot(fig)


if __name__ == "__main__":
    main()


