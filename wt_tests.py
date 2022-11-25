import torch
import cv2
import numpy as np
from swagan import (
    HaarTransform,
    InverseHaarTransform
)

if __name__ == '__main__':
    
    #load an rgb image
    img = cv2.imread('/home/max/Downloads/grg-2Fdamage-2F19288-2Fqkaj710mhgkp0rf1skc33o47b1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # generate a random binary image 
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # generate a randm list of points in the image
    points = []
    for i in range(100):
       points.append((np.random.randint(0, img.shape[0]), np.random.randint(0, img.shape[1])))
    
    # draw the polygon on the mask
    cv2.fillPoly(mask, np.array([points]), 255)
    
    img = mask[:, :, None]

    # convert to torch tensor
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    img = img / 255.0

    # read the number of channels
    channels = img.shape[1]

    # create haar transform
    haar = HaarTransform(1)

    # apply haar transform
    img_haar = haar(img)

    # convert to numpy array
    np_img_haar = img_haar.squeeze(0).permute(1, 2, 0).numpy()

    # using matplotlib display the 12 haar transformed channels in a grid of 3x4 using heatmap as color map
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(channels, 4)
    for i in range(channels):
        for j in range(4):
            axs[j].imshow(np_img_haar[:, :, j*channels+i], cmap='gray')
            axs[j].axis('off')
    plt.show()

    # create inverse haar transform
    inv_haar = InverseHaarTransform(1)

    # apply inverse haar transform
    img_inv_haar = inv_haar(img_haar)

    # convert to numpy array
    np_img_inv_haar = img_inv_haar.squeeze(0).permute(1, 2, 0).numpy()

    # prepare inverse harr transform for visualization
    np_img_inv_haar = np_img_inv_haar * 255.0
    np_img_inv_haar = np_img_inv_haar.astype(np.uint8)

    # visualize inverse haar transform
    cv2.imshow('inv_haar', np_img_inv_haar)

    # wait for key press
    cv2.waitKey(0)