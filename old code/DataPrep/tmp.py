import os
import shutil

fname = "/home/arbellea/Downloads/RPE_Cy5_TexasRed/M20180725_w1Cy5 Camera_s{}_t{}.TIF"
dirname = "/newdisk/arbellea/Data/ISBI/Challenge/Lahav/{:02d}"

base_dir = 2
for s in range(7, 13):
    os.makedirs(dirname.format(base_dir), exist_ok=True)
    outname = os.path.join(dirname.format(base_dir), 't{:03d}.tif')
    base_dir += 1
    for t in range(200):
        if not os.path.exists(fname.format(s, t+1)):
            break
        shutil.copy(fname.format(s,t+1), outname.format(t))
