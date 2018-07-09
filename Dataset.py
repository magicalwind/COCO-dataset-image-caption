
# coding: utf-8

# 
# ## Step 1: Initialize the COCO API
# 
# 

# In[1]:


import os
import sys
sys.path.append('./cocoapi/PythonAPI')
from pycocotools.coco import COCO

# initialize COCO API for instance annotations
dataDir = './cocoapi'
dataType = 'val2014'
instances_annFile = os.path.join(dataDir, 'annotations/instances_{}.json'.format(dataType))
coco = COCO(instances_annFile)

# initialize COCO API for caption annotations
captions_annFile = os.path.join(dataDir, 'annotations/captions_{}.json'.format(dataType))
coco_caps = COCO(captions_annFile)

# get image ids 
ids = list(coco.anns.keys())


# In[2]:


import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[3]:


ann_id = np.random.choice(ids)


# In[4]:


coco.anns[ann_id]


# In[5]:


img_id = coco.anns[ann_id]['image_id']


# In[6]:


import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# pick a random image and obtain the corresponding URL
ann_id = np.random.choice(ids)
img_id = coco.anns[ann_id]['image_id']
img = coco.loadImgs(img_id)[0]
url = img['coco_url']

# print URL and visualize corresponding image
print(url)
I = io.imread(url)
plt.axis('off')
plt.imshow(I)
plt.show()

# load and display captions
annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)

