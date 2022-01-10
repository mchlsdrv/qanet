import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
seq = 3
im_dir = '/newdisk/arbellea/Data/ISBI/Challenge/Lahav/{:02d}/'.format(seq)
seg_dir = '/newdisk/arbellea/DeepCellSegOut/UnetLSTM/outputs/Lahav-02_Challenge_2018-07-30_152822/final_vis/'
seg_dir = '/newdisk/arbellea/DeepCellSegOut/UnetLSTM/outputs/Lahav-03_Challenge_2018-08-01_145952/final_vis/'
seg_fnames = os.path.join(seg_dir,'mask{:03d}.tif')
im_fnames = os.path.join(im_dir,'t{:03d}.tif')
out_dir = seg_dir.replace('final_vis', 'final_vis_example')
save_fnames = os.path.join(out_dir, 't{:03d}.tif')
vid_fnames = os.path.join(out_dir, 'vid.avi')
os.makedirs(out_dir, exist_ok=True)
cm = set([])
cmaps = ['tab20', 'tab20b', 'Set1', 'Set2','Set3', 'Dark2', 'Accent', 'Paired']
for m in cmaps:
 cm = cm.union( set(plt.get_cmap(m).colors))
cm = list(cm)
random.shuffle(cm)
cm = [(c[0]*255, c[1]*255, c[2]*255) for c in cm]

max_colors = len(cm)
out = cv2.VideoWriter(vid_fnames, cv2.VideoWriter_fourcc('M','J','P','G'), 3, (1024,1024))
try:
    for t in range(192):
        I = cv2.imread(im_fnames.format(t),-1)
        S = cv2.imread(seg_fnames.format(t),-1)
        p1 = np.percentile(I, 0.01)
        p2 = np.percentile(I, 99.99)

        I2 = np.uint8(np.minimum(np.maximum((I-p1)/(p2-p1)*255, 0), 255))
        I2 = np.stack([I2]*3, 2)
        for s in np.unique(S):
            if s == 0:
                continue
            bw = S == s
            (_, contours, _) = cv2.findContours(bw.astype(np.uint8).copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(I2, contours, -1, cm[s % max_colors], 2)
        cv2.imwrite(save_fnames.format(t), np.flip(I2,2))
        out.write(I2)
except:
    pass

out.release()