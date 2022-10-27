from skimage import io
import matplotlib.pyplot as plt

img = f'training/mask/polyp2.mp4_0.183806.png_1,182.png'
img = io.imread(img)

plt.imshow(img)
plt.show()