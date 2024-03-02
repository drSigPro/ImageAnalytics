#%% Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit app
st.divider()
st.title("Image Filtering Demo \t")
st.divider()

# Upload an audio file
uploaded_file = st.file_uploader("Upload an image")

# Kernel Size
SmoothWindow = st.sidebar.slider("Select Smoothing Window Size", min_value=3, max_value=255, value=3, step=2)
gradXWindow = st.sidebar.slider("Select Gradient X Window Size", min_value=3, max_value=31, value=3, step=2)
gradYWindow = st.sidebar.slider("Select Gradient Y Window Size", min_value=3, max_value=31, value=3, step=2)
medianWindow = st.sidebar.slider("Select Median Window Size", min_value=3, max_value=255, value=3, step=2)

# Main Streamlit app
def main():
    
    if uploaded_file:

        if SmoothWindow or gradXWindow or gradYWindow or medianWindow:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            
            Img=cv2.imdecode(file_bytes, 1)
            
            if (len(np.shape(Img))==3):
                Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
           
            # Smoothing (Gaussian blur)
            smoothed = cv2.GaussianBlur(Img, (SmoothWindow, SmoothWindow), 0)

            # Image gradients (Sobel)
            gradient_x = cv2.Sobel(Img, cv2.CV_64F, 1, 0, ksize=gradXWindow)
            gradient_x[gradient_x<0] = 0
            gradient_y = cv2.Sobel(Img, cv2.CV_64F, 0, 1, ksize=gradYWindow)
            gradient_y[gradient_y<0] = 0
            
            # Median filter
            median_filtered = cv2.medianBlur(Img, medianWindow)
            
            
            
                      
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs[0,0].imshow(Img,cmap='gray')     
            axs[0,0].axis('off')  
            axs[0,0].set_title('Original Image')
            
            axs[0,1].imshow(smoothed,cmap='gray') 
            axs[0,1].axis('off')  
            axs[0,1].set_title('Smoothed Image')     
            
            axs[1,0].imshow(gradient_x,cmap='gray') 
            axs[1,0].axis('off')  
            axs[1,0].set_title('X-Gradient Image') 
            
            axs[1,1].imshow(gradient_y,cmap='gray') 
            axs[1,1].axis('off')  
            axs[1,1].set_title('Y-Gradient Image') 
            
            st.pyplot(fig)
            
            fig = plt.figure(figsize=(1, 1))
            plt.imshow(median_filtered,cmap='gray')     
            plt.axis('off')  
            plt.title('Median Filtered Image')
            st.pyplot(fig)


if __name__ == "__main__":
    main()


