#%% Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

#%% Function to load and process the uploaded audio file (Assuming image is uint8)
def histeq(Img):
    histImg = cv2.calcHist([Img], [0], None, [256], [0, 256])
    h=histImg/len(Img.ravel())
    C=np.cumsum(h)
    eqImg=C[Img] 
    eqImg = np.uint8(255 * eqImg)
    
    return eqImg

def contrast_stretching(img):
    img=img.astype('float')    
    normImg = (img - np.min(img)) / (np.max(img) - np.min(img))
    normImg = np.uint8(normImg*255)
    return normImg

# Streamlit app
st.divider()
st.title("Histogram Equalization Demo \t")
st.divider()

# Upload an audio file
uploaded_file = st.file_uploader("Upload an image")


# Main Streamlit app
def main():
    
    if uploaded_file:

        if st.button('Histogram Processing'):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            
            Img=cv2.imdecode(file_bytes, 1)
            
            if (len(np.shape(Img))==3):
                Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
           
            # Process the uploaded audio
            normImg = contrast_stretching(Img)
            eqImg = histeq(Img)
            
            # Calculate histogram
            histImg = cv2.calcHist([Img], [0], None, [256], [0, 256])
            histImgCS = cv2.calcHist([normImg], [0], None, [256], [0, 256])
            histImgHEQ = cv2.calcHist([eqImg], [0], None, [256], [0, 256])
            
            
            st.subheader("Original Image")            
            fig, axs = plt.subplots(1, 2, figsize=(10, 3))
            axs[0].imshow(Img,cmap='gray',vmin=0, vmax=255)     
            axs[0].axis('off')        
            
            axs[1].plot(histImg, color='black')
            axs[1].set_xlim([0, 256])
            axs[1].set_ylim([0, np.max(histImg)])
            axs[1].set_xlabel('Pixel Value')
            axs[1].set_ylabel('Frequency')                  
            st.pyplot(fig)
            
            st.subheader("Constrast Stretched Image")
            fig, axs = plt.subplots(1, 2, figsize=(10, 3))
            axs[0].imshow(normImg,cmap='gray')     
            axs[0].axis('off')        
            
            axs[1].plot(histImgCS, color='black')
            axs[1].set_xlim([0, 256])
            axs[1].set_ylim([0, np.max(histImgCS)])
            axs[1].set_xlabel('Pixel Value')
            axs[1].set_ylabel('Frequency')                  
            st.pyplot(fig)
            
            st.subheader("Histogram Equalized Image")
            fig, axs = plt.subplots(1, 2, figsize=(10, 3))
            axs[0].imshow(eqImg,cmap='gray')     
            axs[0].axis('off')        
            
            axs[1].plot(histImgHEQ, color='black')
            axs[1].set_xlim([0, 256])
            axs[1].set_ylim([0, np.max(histImgHEQ)])
            axs[1].set_xlabel('Pixel Value')
            axs[1].set_ylabel('Frequency')                  
            st.pyplot(fig)


if __name__ == "__main__":
    main()


