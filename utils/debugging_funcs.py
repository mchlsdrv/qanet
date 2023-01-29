import matplotlib.pyplot as plt


def show_image_mask(image, mask, name: str = 'testimg'):
    fig, ax = plt.subplots(1, 2, figsize=(10, 20))
    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(mask, cmap='gray')
    fig.savefig(f'{name}.png')
    plt.show()
