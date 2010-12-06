#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os


ROOT = '../data/'

from starflow.utils import MakeDir, activate

def initialize(creates = ROOT):
    MakeDir(creates)

@activate(lambda x : x[0],lambda x : x[1])
def wget(getpath,savepath,opstring=''):
    os.system('wget ' + opstring + ' "' + getpath + '" -O "' + savepath + '"')
    
CALTECH_101_URL = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz'
CALTECH_101 = os.path.join(ROOT , '101_ObjectCategories/')
   
def get_caltech_101(depends_on = CALTECH_101_URL,creates=os.path.join(ROOT , '101_ObjectCategories.tar.gz')):
    wget(depends_on,creates)
    
def unzip_caltech_101(depends_on = os.path.join(ROOT , '101_ObjectCategories.tar.gz'),creates=CALTECH_101):
    os.system('cd ' + ROOT + ' && tar xzvf 101_ObjectCategories.tar.gz')
    


