#%% Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

#%% Function to load and process the uploaded audio file (Assuming image is uint8)
def quantization(Img256,bits):
    level = np.power(2,8)/np.power(2,bits)
    
    #%% Reducing the intensity resolution
    Imgq = (np.ceil(Img256 / level) * level - 1) # Reducing from 8bits to 1bit
    Imgq = np.uint8(Imgq)
    return Imgq

#%% Function to load and process the uploaded audio file (Assuming image is uint8)
def resampling(Img,scale_percent):
    
    scale_percent = 1/scale_percent
    #%% Reducing the spatial resolution
    width = int(Img.shape[1] * scale_percent )
    height = int(Img.shape[0] * scale_percent)
    dim = (width, height)
    Imgs = cv2.resize(Img,dim,interpolation = cv2.INTER_AREA)
    
    return Imgs

# Streamlit app
st.divider()
st.title("Image resampling and Quantization Demo \t")
st.divider()

# Upload an audio file
uploaded_file = st.file_uploader("Upload an image")

# Sampling rate and quantization levels
sampling_rate = st.sidebar.slider("Select Reduction Size", min_value=1, max_value=50, value=1)
quantization_levels = st.sidebar.slider("Select quantization (bits)", min_value=1, max_value=8, value=8)

# Main Streamlit app
def main():
    
    if uploaded_file:

        if sampling_rate or quantization_levels:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            
            Img=cv2.imdecode(file_bytes, 1)
            
            if (len(np.shape(Img))==3):
                Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
           
            # Process the uploaded audio
            Imgs = resampling(Img,sampling_rate)
            Imgq = quantization(Img,quantization_levels)
            
            # Calculate histogram
            histImg = cv2.calcHist([Img], [0], None, [256], [0, 256])
            histImgq = cv2.calcHist([Imgq], [0], None, [256], [0, 256])
            histImgs = cv2.calcHist([Imgs], [0], None, [256], [0, 256])
            
            
            st.subheader("Original Image")            
            fig, axs = plt.subplots(1, 2, figsize=(10, 3))
            axs[0].imshow(Img,cmap='gray')     
            axs[0].axis('off')        
            
            axs[1].plot(histImg, color='black')
            axs[1].set_xlim([0, 256])
            axs[1].set_ylim([0, np.max(histImg)])
            axs[1].set_xlabel('Pixel Value')
            axs[1].set_ylabel('Frequency')                  
            st.pyplot(fig)
            
            st.subheader("Sampled Image")
            fig, axs = plt.subplots(1, 2, figsize=(10, 3))
            axs[0].imshow(Imgs,cmap='gray')     
            axs[0].axis('off')        
            
            axs[1].plot(histImgs, color='black')
            axs[1].set_xlim([0, 256])
            axs[1].set_ylim([0, np.max(histImgs)])
            axs[1].set_xlabel('Pixel Value')
            axs[1].set_ylabel('Frequency')                  
            st.pyplot(fig)
            
            st.subheader("Quantized Image")
            fig, axs = plt.subplots(1, 2, figsize=(10, 3))
            axs[0].imshow(Imgq,cmap='gray')     
            axs[0].axis('off')        
            
            axs[1].plot(histImgq, color='black')
            axs[1].set_xlim([0, 256])
            axs[1].set_ylim([0, np.max(histImgq)])
            axs[1].set_xlabel('Pixel Value')
            axs[1].set_ylabel('Frequency')                  
            st.pyplot(fig)


if __name__ == "__main__":
    main()


