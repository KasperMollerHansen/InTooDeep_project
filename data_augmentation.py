#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
#%%

path1 = "data/camera_01/"
path2 = "data/camera_02/"
img_list1 = os.listdir(path1)
img_list2 = os.listdir(path2)
images = []
with open("data/rotations_w_images.csv", "r") as f:
    reader = csv.reader(f)
    for line in reader:
        images.append(line)
images
N = len(images)
#%%
r_idx = np.random.randint(1, N)
img1 = plt.imread(path1+images[r_idx][2])
img2 = plt.imread(path2+images[r_idx][3])
fig, ax = plt.subplots(1,2,figsize=(12,5))
ax[0].imshow(img1)
ax[1].imshow(img2)
fig.suptitle(f"Base: {np.round(float(images[r_idx][0]),2)}, Blades: {np.round(float(images[r_idx][1]),2)}")
fig.tight_layout()
plt.show()

 
#%%

t = np.array([1,2,2])
t2 = np.array([3,2,1])
diff = t-t2
print(np.mean(np.sum(diff)**2))
print(np.mean(np.sum(diff**2)))