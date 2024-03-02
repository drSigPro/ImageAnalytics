#%% Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def corner_response(image, k, smooth_flag,kSize):
    
    # Sobel kernels
    Sx = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]])

    Sy = Sx.T

    # Gaussian Kernel
    G = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]])/16
    
    if (smooth_flag):
        image = cv2.GaussianBlur(image, (kSize, kSize), 0)
    
    # compute first derivatives
    dx = cv2.filter2D(image, ddepth=-1, kernel=Sx)
    dy = cv2.filter2D(image, ddepth=-1, kernel=Sy)

    # Gaussian Filter
    A = cv2.filter2D(dx*dx, ddepth=-1, kernel=G)
    B = cv2.filter2D(dy*dy, ddepth=-1, kernel=G)
    C = cv2.filter2D(dx*dy, ddepth=-1, kernel=G)

    # compute corner response at all pixels
    R = (A*B - (C*C)) - k*(A + B)*(A + B)
    return R, A, B, C


def get_harris_corners(image, k,smooth_flag,kSize):

    # compute corner response
    R, A, B, C = corner_response(image, k,smooth_flag,kSize)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(R > 1e-2))
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    
    corners = cv2.cornerSubPix(image, np.float32(centroids), (9,9), (-1,-1), criteria)
    return corners, A, B, C

# Streamlit app
st.divider()
st.title("Corner Detection Demo \t")
st.divider()

# Upload an audio file
uploaded_file = st.file_uploader("Upload an image")

# Hyper Parameters
k = st.sidebar.slider("Select Regularization Parameter", min_value=0.000, max_value=1.000, value=0.050, step=0.005)
kSize = st.sidebar.slider("Select Kernel Size", min_value=3, max_value=51, value=3, step=2)
smooth_flag = st.sidebar.checkbox("Enable Smoothing", value=False)

# Main Streamlit app
def main():
    
    if uploaded_file:
        if k or kSize or smooth_flag:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            
            Img8=cv2.imdecode(file_bytes, 1)
            
          
            if (len(np.shape(Img8))==3):
                Img8 = cv2.cvtColor(Img8, cv2.COLOR_BGR2GRAY)
            
            Img = np.float32(Img8)
            Img /= Img.max()

            corners, A, B, C = get_harris_corners(Img,k,smooth_flag,kSize)            
            
            image_out = np.dstack((Img, Img, Img))
            for (x, y) in corners:
                x = np.round(x).astype(int)
                y = np.round(y).astype(int)
                cv2.circle(image_out, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
            
            # Display the results
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            
            # Original Image
            axs[0, 0].imshow(Img, cmap='gray')
            axs[0, 0].set_title('Original Image')
            axs[0, 0].axis('off')
            
            # Gaussian Blur
            axs[0, 1].imshow(A, cmap='gray')
            axs[0, 1].set_title('dx x dx')
            axs[0, 1].axis('off')
            
            # Gradient Magnitude
            axs[1, 0].imshow(B, cmap='gray')
            axs[1, 0].set_title('dy x dy')
            axs[1, 0].axis('off')
            
            # Gradient Direction
            axs[1, 1].imshow(C, cmap='gray')
            axs[1, 1].set_title('dx x dy')
            axs[1, 1].axis('off')
            
            st.pyplot(fig)
            
            fig = plt.figure(figsize=(1, 1))
            plt.imshow(image_out,cmap='gray')     
            plt.axis('off')  
            plt.title('Image with Corners')
            st.pyplot(fig)


if __name__ == "__main__":
    main()


