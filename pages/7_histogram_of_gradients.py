# Reference: https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html

#%% Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from skimage.feature import hog
from skimage import exposure

def histeq(Img):
    histImg = cv2.calcHist([Img], [0], None, [256], [0, 256])
    h=histImg/len(Img.ravel())
    C=np.cumsum(h)
    eqImg=C[Img] 
    eqImg = np.uint8(255 * eqImg)
    
    return eqImg


# Streamlit app
st.divider()
st.title("Histogram of Gradients Demo \t")
st.divider()
st.markdown(
    """
    *Reference: https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html*
    
    """
)
# Upload an audio file
uploaded_file = st.file_uploader("Upload an image")

# Kernel Size
orientations = st.sidebar.slider("Select Number of Orientations", min_value=2, max_value=24, value=8, step = 1)
pixels_per_cell = st.sidebar.slider("Select Number of Pixels per cell", min_value=4, max_value=64, value=8, step=1)
cells_per_block = st.sidebar.slider("Select Number of Pixels per cell", min_value=1, max_value=7, value=1, step=1)
hist_eq = st.sidebar.checkbox("Enable Histogram Equalization", value=False)
METHOD = 'uniform'


# Main Streamlit app
def main():
    
    if uploaded_file:

        if orientations or pixels_per_cell or cells_per_block or hist_eq:
            
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            
            Img=cv2.imdecode(file_bytes, 1)
            
            if (len(np.shape(Img))==1):
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
            
            if (hist_eq):
                Img = histeq(Img)
           
            
            fd, hog_image = hog(Img, orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                                cells_per_block=(cells_per_block, cells_per_block), visualize=True, channel_axis=-1)
            
            # fd, hog_image = hog(Img, orientations=8, pixels_per_cell=(16, 16),
            #         cells_per_block=(1, 1), visualize=True, channel_axis=-1)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            
            ax1.axis('off')
            ax1.imshow(Img, cmap=plt.cm.gray)
            ax1.set_title('Input image')
            
            # Rescale histogram for better display
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            
            ax2.axis('off')
            ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            ax2.set_title('Histogram of Oriented Gradients')
            plt.show()
            
            st.pyplot(fig)


if __name__ == "__main__":
    main()


