#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

from starflow.protocols import actualize, protocolize
from starflow.utils import MakeDir, activate

import v1.v1s as v1s



# -----------------------------------------------------------------------------

def v1s_protocol(param_fname, img_path, results_dir,write=False):
    param_path = os.path.abspath(param_fname)
    v1s_params = {}
    execfile(param_path, {}, v1s_params)
    
    ntrials = v1s_params['protocol']['ntrials']

    D = [('initialize',MakeDir,(results_dir,))]
 
    D += [('run_' + str(i),v1s.run_one_trial,(param_fname, img_path, os.path.join(results_dir,'results_' + str(i) + '.pickle'))) for i in range(ntrials)]
        
    if write:
        actualize(D)
    return D
   
COMP_ROOT = '../v1s_computed_data/'

def initialize(creates = COMP_ROOT):
    MakeDir(COMP_ROOT)


@protocolize()
def v1s_simple(depends_on = ('../v1/params_simple.py','../data/101_ObjectCategories')):
    D = v1s_protocol(depends_on[0],depends_on[1],os.path.join(COMP_ROOT,'simple/'))
    actualize(D)
    
    
    