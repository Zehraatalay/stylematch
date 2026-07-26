import cv2
import pickle
import matplotlib.pyplot as plt 
import argparse
from detect_eval import detect_image 

CLASS_NAMES = {0 : "no object"}
PROB_THRESH = 0.20 

def _pt(x,y) : 
    return int(round(float(x))), int(round(float(y)))

