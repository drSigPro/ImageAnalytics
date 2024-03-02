# Reference: https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html

#%% Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from skimage import feature
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage.color import label2rgb


def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')

def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')

def contrast_stretching(img):
    img=img.astype('float')    
    normImg = (img - np.min(img)) / (np.max(img) - np.min(img))
    normImg = np.uint8(normImg*255)
    return normImg

def histeq(Img):
    histImg = cv2.calcHist([Img], [0], None, [256], [0, 256])
    h=histImg/len(Img.ravel())
    C=np.cumsum(h)
    eqImg=C[Img] 
    eqImg = np.uint8(255 * eqImg)
    
    return eqImg


# Streamlit app
st.divider()
st.title("Local Binary Patterns Demo \t")
st.divider()

st.markdown(
    """
    *Reference: https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html*
    
    """
)

# Upload an audio file
uploaded_file = st.file_uploader("Upload an image")

# Kernel Size
radius = st.sidebar.slider("Select Radius Size", min_value=3, max_value=16, value=3, step = 1)
num_points = 8* radius #st.sidebar.slider("Select Number of Pixels in the Radius", min_value=4, max_value=256, value=24, step=1)
constrast_stretch = st.sidebar.checkbox("Enable Contrast Stretch", value=False)
hist_eq = st.sidebar.checkbox("Enable Histogram Equalization", value=False)
METHOD = 'uniform'


# Main Streamlit app
def main():
    
    if uploaded_file:

        if num_points or radius or constrast_stretch or hist_eq:
            
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            
            Img=cv2.imdecode(file_bytes, 1)
            
            if (len(np.shape(Img))==3):
                Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
            
            if (constrast_stretch):
                Img = contrast_stretching(Img)
            
            if (hist_eq):
                Img = histeq(Img)
           
            lbp = local_binary_pattern(Img, num_points, radius, METHOD)
            
            fig = plt.figure(figsize=(1, 1))
            plt.imshow(Img,cmap='gray',vmin=0, vmax=255)     
            plt.axis('off')  
            plt.title('Original Image')
            st.pyplot(fig)
            
            # plot histograms of LBP of textures
            fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
            plt.gray()
            
            titles = ('edge', 'flat', 'corner')
            w  = radius - 1
            edge_labels = range(num_points // 2 - w, num_points // 2 + w + 1)
            flat_labels = list(range(0, w + 1)) + list(range(num_points - w, num_points + 2))
            i_14 = num_points // 4            # 1/4th of the histogram
            i_34 = 3 * (num_points // 4)      # 3/4th of the histogram
            corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                             list(range(i_34 - w, i_34 + w + 1)))
            
            label_sets = (edge_labels, flat_labels, corner_labels)
            
            for ax, labels in zip(ax_img, label_sets):
                ax.imshow(overlay_labels(Img, lbp, labels))
            
            for ax, labels, name in zip(ax_hist, label_sets, titles):
                counts, _, bars = hist(ax, lbp)
                highlight_bars(bars, labels)
                ax.set_ylim(top=np.max(counts[:-1]))
                ax.set_xlim(right=num_points + 2)
                ax.set_title(name)
            
            ax_hist[0].set_ylabel('Percentage')
            for ax in ax_img:
                ax.axis('off')
            
            st.pyplot(fig)


if __name__ == "__main__":
    main()


