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
def v1s_simple(depends_on = '../v1/params_simple.py'):
    D = v1s_protocol(depends_on,'../data/101_ObjectCategories',os.path.join(COMP_ROOT,'simple/'))
    actualize(D)
    
  
def v1s_seq_protocol(param_fname, img_path, results_dir, aggregate_file, write=False):
    param_path = os.path.abspath(param_fname)
    v1s_params = {}
    execfile(param_path, {}, v1s_params)
    
    ntrials = v1s_params['protocol']['ntrials']

    D = [('initialize',MakeDir,(results_dir,))]
 
    for i in range(ntrials):
        D += v1s.trial_protocol(param_fname, img_path, results_dir, prefix = 'trial_' + str(i), make_container = False)
        
    #D += [('aggregate_results', v1s.aggregate_results, (results_dir, aggregate_file))]
    
       
    if write:
        actualize(D)
        
    return D  
    

@protocolize()
def v1s_simple_seq(depends_on = '../v1/params_simple.py'):
    D = v1s_seq_protocol(depends_on,'../data/101_ObjectCategories',os.path.join(COMP_ROOT,'simple/'),os.path.join(COMP_ROOT,'simple_results.pickle'))
    actualize(D)
    