import matplotlib.pyplot as plt
import numpy as np

def visualize(fig_name, path, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    if path is not None:
        plt.savefig(f"{path}/results/figures/{fig_name}")
        plt.clf()
    else:
        plt.show()
        #plt.clf()

def visualize_edges(fig_name, path, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    if path is not None:
        plt.savefig(f"{path}/results_for_edges/figures/{fig_name}")
        plt.clf()
    else:
        plt.show()
        #plt.clf()

# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x