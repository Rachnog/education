import numpy as np
import os
import os.path
import pandas as pd

import PIL
from skimage.feature import hog
from skimage import data, color, exposure
from skimage import io
import dlib

W = 50

# DETECTOR = dlib.get_frontal_face_detector() # uncomment to detect faces


def detect_face(img):
    '''
        Detect faces on an image and return cropped one with the face only
    '''
    dets, scores, idx = DETECTOR.run(img, 1, -1)
    if len(dets) > 0:
    	d = dets[0]
    	crop = img[d.top():d.bottom(), d.left():d.right()]	
        print 'Face cropped!'
        return crop
    else:
	    print 'Face wasn\'t cropped'
	    return img


def from_text_to_matrix(data):
	res = []
	for line in data:
		try:
			splitted = line.strip().split(' ')
			x = float(splitted[0])
			y = float(splitted[-1])
			res.append([x, y])
		except:
			continue
	return res


def load_photos_from_folders(filepath):
	points = {}
	for dirpath, dirnames, filenames in os.walk(filepath):
	    dirnames.sort()
	    filenames.sort()
	    for filename in [f for f in filenames if f.endswith(".png")]:

	        full_path = os.path.join(dirpath, filename)
	        filename = full_path.split('/')[-1].split('_')
	        person = filename[0]
	        person_movie = filename[1]
	        person_movie_frame = filename[2]
	        name = "%s_%s" % (person, person_movie)
	        if name not in points.keys():
	        	points[name] = []

	        image = io.imread(full_path)

	        # image = detect_face(image)
	        # io.imsave(full_path, image)

	        image = color.rgb2gray(image)
	        image.resize((W, W))

	        points[name].append(image)

	        print filename, 'Loaded'

	return points


def load_keypoints_txt_from_folders(filepath):
	points = {}
	for dirpath, dirnames, filenames in os.walk(filepath):
	    dirnames.sort()
	    filenames.sort()
	    for filename in [f for f in filenames if f.endswith(".txt")]:
	        full_path = os.path.join(dirpath, filename)
	        filename = full_path.split('/')[-1].split('_')
	        person = filename[0]
	        person_movie = filename[1]
	        person_movie_frame = filename[2]
	        name = "%s_%s" % (person, person_movie)
	        if name not in points.keys():
	        	points[name] = []
	        points[name].append(from_text_to_matrix(open(full_path).readlines()))
	return points



def load_label_txt_from_folders(filepath):
	points = {}
	for dirpath, dirnames, filenames in os.walk(filepath):
	    for filename in [f for f in filenames if f.endswith(".txt")]:
	        full_path = os.path.join(dirpath, filename)
	        filename = full_path.split('/')[-1].split('_')
	        person = filename[0]
	        person_movie = filename[1]	        
	        person_movie_frame = filename[2]
	        points["%s_%s" % (person, person_movie)] = from_text_to_matrix(open(full_path).readlines())
	return points