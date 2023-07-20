# move images named "-1" to output_path
"""
import glob
import os

root = '/storage/sjhwang/reid/market1501/Annotation'

out = '/storage/sjhwang/reid/market1501/test'
query = out + '/query'
test = out + '/bounding_box_test'
train = out + '/bounding_box_train'

text = '/*.txt'
t_q = glob.glob(root + '/query' + text)
t_te = glob.glob(root + '/test' + text)
t_tr = glob.glob(root + '/train' + text)

for i in t_q:
    os.system(f"cp {i} {query}")

for i in t_te:
    os.system(f"cp {i} {test}")

for i in t_tr:
    os.system(f"cp {i} {train}")
"""
"""
# change the image name.
import glob
import os

root = '/storage/sjhwang/reid/market1501/test'
query = root + '/query'
test = root + '/bounding_box_test'
train = root + '/bounding_box_train'
img_root = '/*.jpg'

q = glob.glob(query + img_root)
te = glob.glob(test + img_root)
tr = glob.glob(train + img_root)

for i in q:
    if ".jpg.jpg" in i:
        change = i.replace(".jpg.jpg", ".jpg")
        os.system(f'mv {i} {change}')
for i in te:
    if ".jpg.jpg" in i:
        change = i.replace(".jpg.jpg", ".jpg")
        os.system(f'mv {i} {change}')
for i in tr:
    if ".jpg.jpg" in i:
        change = i.replace(".jpg.jpg", ".jpg")
        os.system(f'mv {i} {change}')
"""

import glob
import os

# original folder
root = '/storage/sjhwang/TransReID-SSL/data/market1501'
gt_bbox = root +  '/gt_bbox'
gt_query = root +  '/gt_query'
readme = root +  '/readme.txt'
query = root +  '/query'
train = root +  '/bounding_box_train'
save = '/storage/sjhwang/target_all'
# Makelist
anonys = ['/L_blur/market1501', '/G_blur_sig_7/market1501', '/Blank/market1501', '/Noise_sig_45/market1501', '/Super/market1501', '/Edge/market1501']
deep = ['/deep/market1501']
f_an = [save + i for i in deep]

for i in f_an:
    os.system(f'cp -R {gt_bbox} {i}')
    os.system(f'cp -R {gt_query} {i}')
    os.system(f'cp {readme} {i}')
    os.system(f'cp -R {query} {i}')
    os.system(f'cp -R {train} {i}')

"""
import glob
import os

# original folder
root = '/storage/sjhwang/reid/market1501'
ori = '/test'
gt_bbox = root + ori + '/gt_bbox'
gt_query = root + ori + '/gt_query'
readme = root + ori + '/readme.txt'

# Makelist
anonys = ['/Edge', '/G_blur', '/Noise', '/Blank', '/L_blur', '/Super']
f_an = [root + i for i in anonys]

# output
out = root + '/final'
final = [out+i for i in anonys]


for i, path in enumerate(f_an):
    os.system(f'cp -R {path} {out}')
"""
"""
import glob
import os

# original folder
root = '/storage/sjhwang/reid/market1501'
save = '/storage/sjhwang/reid/result'
ori = '/test'

gt_bbox = root + ori + '/gt_bbox'
gt_query = root + ori + '/gt_query'
readme = root + ori + '/readme.txt'

# Makelist
anonys = ['/Edge', '/G_blur', '/Noise', '/Blank', '/L_blur', '/Super']
f_an = [save + ori + i for i in anonys]

for i in f_an:
    os.system(f'cp -R {query} {i}')

# move to the image 
"""