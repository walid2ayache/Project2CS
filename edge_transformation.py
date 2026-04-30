import numpy as np
import cv2
import os
import time
import glob
import argparse

img_size=64
blur_ksize=3

parser = argparse.ArgumentParser(description='Apply edge detection to ECG images')
parser.add_argument('--source_base', type=str, required=True,
                    help='Base directory of source images (with class subfolders)')
parser.add_argument('--dest_base', type=str, default='/kaggle/working/ECG_edges/Prewitt_v2',
                    help='Base directory to save edge-detected images')

ECG_CLASSES = ['NOR', 'PVC', 'PAB', 'LBB', 'RBB', 'APC', 'VFW', 'VEB']


def Prewitt_v2(image):
    print("reading file---> " + str(image))
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (img_size, img_size))
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=int)
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=int)
    x = cv2.filter2D(image, cv2.CV_16S, kernelx)
    y = cv2.filter2D(image, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    result = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return result


def converter_Prewitt_v2(sourcedir, destdir):
    print("\n\n---reading directory " + sourcedir + "---\n")
    filecnt = 1
    for filename in glob.glob(sourcedir + '/*'):
        imagemat = Prewitt_v2(filename)
        cv2.imwrite(destdir+'/img-'+str(filecnt)+'.png', imagemat)
        filecnt += 1
    print("\n\n--saved in " + destdir + "--\n")


if __name__ == '__main__':
    args = parser.parse_args()
    start = time.time()

    for cls in ECG_CLASSES:
        sourcedir = os.path.join(args.source_base, cls)
        destdir = os.path.join(args.dest_base, cls)

        if not os.path.exists(sourcedir):
            print(f"WARNING: Source not found, skipping: {sourcedir}")
            continue

        os.makedirs(destdir, exist_ok=True)
        print(f"Processing class: {cls}")
        converter_Prewitt_v2(sourcedir, destdir)

    end = time.time()
    print("Total time (min):", (end - start)/60)
    print("=======End edge transformation======")
