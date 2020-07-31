# Converting .csv files to images to train CNN
# Researchers: Jeffrey Chau, Apala Thakur

import os
import numpy as np
import pandas as pd
from PIL import Image
import math

base_dir = './'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_drowsy_dir = os.path.join(train_dir,'drowsy')
train_focus_dir = os.path.join(train_dir,'focus')
train_defocus_dir = os.path.join(train_dir,'defocus')

validation_drowsy_dir = os.path.join(validation_dir,'drowsy')
validation_focus_dir = os.path.join(validation_dir,'focus')
validation_defocus_dir = os.path.join(validation_dir,'defocus')

dirs = [train_drowsy_dir, train_focus_dir, train_defocus_dir, 
validation_drowsy_dir, validation_focus_dir, validation_defocus_dir]
cols = [i for i in range(1,25)]
for dir in dirs:
	print(dir)
	files = os.listdir(dir)
	for fname in files:
		filepath = os.path.join(dir, fname)
		if 'jpg' in filepath:
			continue
		
		f = pd.read_csv(filepath, usecols=cols).values.flatten()
		l = len(f)
		h = math.floor(math.sqrt(l))
		w = math.floor(l/h)
		
		ext = l - (h * w)
		new_len = l - ext
		
		clipped = f[:new_len]
		clipped = clipped.reshape(h,w)
		im = Image.fromarray(clipped, 'RGB')
		s = w if w <= h else h
		im = im.resize((s, s))
		
		image_file = fname.replace('csv', 'jpg')
		image_path = os.path.join(dir, image_file)
		im.save(image_path)


