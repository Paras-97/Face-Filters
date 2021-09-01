import cv2
import matplotlib.pyplot as plt

def filter_prep(val):

    # For puppy filter
    if val == 1:
        neon = cv2.imread('website/static/puppy.png')
        #Getting the shape of the filter image
        original_neon_h, original_neon_w, neon_channels = neon.shape 
        
        # Converting to grayscale
        neon_gray  = cv2.cvtColor(neon, cv2.COLOR_BGR2GRAY)
        
        # create mask and inverse mask of filter image
        ret, original_mask = cv2.threshold(neon_gray, 96, 255, cv2.THRESH_BINARY_INV)
        original_mask_inv = cv2.bitwise_not(original_mask)

        return (neon,original_neon_h, original_neon_w, original_mask, original_mask_inv)
    
    # For Neon_mask filter
    if val == 2:
        neon = cv2.imread('website/static/Neon_mask.png')
        #Getting the shape of the filter image
        original_neon_h, original_neon_w, neon_channels = neon.shape 
        
        # Converting to grayscale
        neon_gray  = cv2.cvtColor(neon, cv2.COLOR_BGR2GRAY)
        
        # create mask and inverse mask of filter image
        ret, original_mask = cv2.threshold(neon_gray, 145, 255, cv2.THRESH_BINARY_INV)
        original_mask_inv = cv2.bitwise_not(original_mask)

        return (neon,original_neon_h, original_neon_w, original_mask, original_mask_inv)
    
    # For witch filter
    if val == 3:
        neon = cv2.imread('website/static/witch.png')
        #Getting the shape of the filter image
        original_neon_h, original_neon_w, neon_channels = neon.shape 
        
        # Converting to grayscale
        neon_gray  = cv2.cvtColor(neon, cv2.COLOR_BGR2GRAY)
        
        # create mask and inverse mask of filter image
        ret, original_mask = cv2.threshold(neon_gray, 40, 255, cv2.THRESH_BINARY_INV)
        original_mask_inv = cv2.bitwise_not(original_mask)

        return (neon,original_neon_h, original_neon_w, original_mask, original_mask_inv)
    





