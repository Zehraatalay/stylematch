import cv2
import pickle
import matplotlib.pyplot as plt 
import argparse
from detect_eval import detect_image 

CLASS_NAMES = {0 : "no object"}
PROB_THRESH = 0.20 

def _pt(x,y) : 
    return int(round(float(x))), int(round(float(y)))

def parse_args() : 
    p = argparse.ArgumentParser()
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--margin", type=float, default=0.008)
    p.add_argument("--min-object-prob", type=float, default=0.20)
    p.add_argument("--n", type=int, default=3)
    return p.parse_args()