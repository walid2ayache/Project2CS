import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import glob


img_size = 512

def Resize(image):
    image = Image.open(image).convert('L')
    image = image.thumbnail((img_size, img_size))
    return image


def Resize_images(sourcedir, destdir):
    print("\n\n---reading directory " + sourcedir + "---\n")
    filecnt = 1
    for filename in glob.glob(sourcedir + '/*'):
        image = Resize(filename)
        image.save(os.path.join(destdir, f"{filecnt}.png"))
        filecnt += 1
    print("\n\n--saved in " + destdir + "--\n")


if __name__ == '__main__':
    # ============================================================
    # KAGGLE PATHS
    # ============================================================
    sourcedir = '/kaggle/working/ECG_images/train/NOR'  # Change per class
    destdir = '/kaggle/working/ECG_resized/NOR'          # Change per class
    os.makedirs(destdir, exist_ok=True)
    print("The new directory is created!")
    Resize_images(sourcedir, destdir)
