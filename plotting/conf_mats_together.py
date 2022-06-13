import matplotlib
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

source = "/path/to/result/images/directory/"    ## TODO: update path variable here ##

im0 = plt.imread(source + ".png")
im1 = plt.imread(source + "_blankCTAbl_0.png")
im2 = plt.imread(source + "_GNNAbl.png")
im3 = plt.imread(source + "_noPre.png")

ims = [im0,im1,im2,im3]
for imsdx, im in enumerate(ims):
    ims[imsdx] = im[40:560]
im0, im1, im2, im3 = ims

upper = np.concatenate((im0,im1), axis=1)
lower = np.concatenate((im2,im3), axis=1)
whole = np.concatenate((upper, lower), axis=0)

plt.text(20, 20, "a)", c='k', fontweight="bold")
plt.text(620, 20, "b)", c='k', fontweight="bold")
plt.text(20, 540, "c)", c='k', fontweight="bold")
plt.text(620, 540, "d)", c='k', fontweight="bold")

plt.imshow(whole)
plt.axis('off')
plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0.2,wspace=0.1)
plt.show()