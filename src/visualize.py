import matplotlib.pyplot as plt

def visualize(images, epoch):
    plt.figure(figsize=(16,4))
    for i in range(8):
        plt.subplot(1,8,i+1)
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
    print(f"Epoch: {epoch+1}")
    plt.show()