#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import cPickle as pickle

from starflow.utils import activate

from itertools import izip, count

import scipy
newaxis = scipy.newaxis

import PyML
if PyML.__version__ < "0.6.14":
    print "PyML 0.6.14 or later required"
    raise SystemExit
if PyML.__version__ >= "0.7.0":
    from PyML.classifiers import multi 

import v1.v1s_math as v1s_math
import v1.v1s_funcs as v1s_funcs


EXTENSIONS = ['.png', '.jpg']

conv = scipy.signal.convolve

@activate(lambda x : (x[0],x[1]), lambda x : x[2])
def run_one_trial(param_fname,img_path,result_file):
    
    V = V1S(param_fname, img_path)
    
    param_path = os.path.abspath(param_fname)
    v1s_params = {}
    execfile(param_path, {}, v1s_params)

    model = v1s_params['model']
    pca_threshold = v1s_params['pca_threshold']
    
    r = V._run_one_trial(model,pca_threshold)
    
    out_file = open(result_file,'w')
    pickle.dump(r,out_file)
    


# -----------------------------------------------------------------------------
class V1S(object):

    # -------------------------------------------------------------------------
    def __init__(self, param_fname, img_path):
    
        # -- get parameters
        param_path = os.path.abspath(param_fname)
        v1s_params = {}
        execfile(param_path, {}, v1s_params)
    
        # -- get image filenames
        img_path = os.path.abspath(img_path)
        
        # navigate tree structure and collect a list of files to process
        if not os.path.isdir(img_path):
            raise ValueError, "%s is not a directory" % (img_path)
        tree = os.walk(img_path)
        filelist = []
        categories = tree.next()[1]    
        for root, dirs, files in tree:
            if dirs != []:
                msgs = ["invalid image tree structure:"]
                for d in dirs:
                    msgs += ["  "+"/".join([root, d])]
                msg = "\n".join(msgs)
                raise Exception, msg
            filelist += [ root+'/'+f for f in files if os.path.splitext(f)[-1] in EXTENSIONS ]
        filelist.sort()    
        
        kwargs = v1s_params['protocol']
        kwargs['filelist'] = filelist
    
        self.__dict__.update(kwargs)
        self._cache = {}
        self._labels = None
        self._filt_l = None


    def _run_one_trial(self,model,pca_threshold):
        filelists_dict = self._shuffle_images()
        
        # -- Get training examples
        #   (i.e. read in the images, apply the model, and construct a 
        #    feature vector)
        #   The first ntrain examples (of the shuffled list) will be used for
        #   training.
        print "="*80
        print "Training..."
        print "="*80        
        ntrain = self.ntrain
        train_fvectors, train_flabels, train_fnames = \
                        self._get_fvectors_flabels_fnames(filelists_dict,
                                                          model, 0, ntrain)                                           
        print "training samples:", len(train_fnames)
        
        #sphere the data
        train_fvectors, v_sub, v_div = self._sphere(train_fvectors)
        
        #reduce dimensions
        train_fvectors, eigvectors = self._dimr(train_fvectors,pca_threshold)
        
        #train SVM
        classifier = self._train_svm(train_fvectors,train_flabels,filelists_dict) 
        
        #get test features
        print "="*80
        print "Testing..."
        print "="*80 
        
        #get hook
        hook_params = {'v_sub': v_sub, 'v_div' : v_div, 'eigvectors' : eigvectors}
        hook = self._get_hook(train_fvectors,**hook_params) 
         
        # -- Get testing examples
        ntest = self.ntest    
        test_fvectors, test_flabels, test_fnames = \
                       self._get_fvectors_flabels_fnames(filelists_dict, model,
                                                         ntrain, ntrain+ntest,
                                                         hook)
        print "testing samples:", len(test_fnames)
        
        #safety check
        self._safety_check(test_fnames,train_fnames)
        
        #test svm
        results = self._test_svm(test_fvectors,test_flabels,classifier)
        return results
        
  
    def _get_hook(self, train_fvectors, **train_params):
    
        nvectors , vsize = train_fvectors.shape
        
        v_sub = train_params['v_sub']
        v_div = train_params['v_div']
        eigvectors = train_params['eigvectors']
            
        if nvectors < vsize:
            hook = lambda vector: scipy.dot(((vector-v_sub) / v_div), eigvectors)
        else:
            hook = lambda vector: ((vector-v_sub) / v_div)
            
        return hook
            
        
    def _test_svm(self,test_fvectors,test_flabels,clas):    
        test_data = self._get_sparse_data(test_fvectors,test_flabels)
        
        self._labels = clas.labels.classLabels
        
        print "testing..."
        results = clas.test(test_data)
        results.computeStats()
    
        return results
 
 
    def _shuffle_images(self):
    
        filelist = self.filelist
         
        # -- Organize images into the appropriate categories
        cats = {}
        for f in filelist:
            cat = "/".join(f.split('/')[:-1])
            name = f.split('/')[-1]
            if cat not in cats:
                cats[cat] = [name]
            else:
                cats[cat] += [name]
    
        # -- Shuffle the images into a new random order
        filelists_dict = {}
        for cat in cats:
            filelist = cats[cat]
            scipy.random.seed(self.seed)
            scipy.random.shuffle(filelist)
            self.seed += 1
            filelist = [ cat + '/' + f for f in filelist ]
            filelists_dict[cat] = filelist
            
        return filelists_dict

    def _sphere(self,fvectors):  
    
        # -- Sphere the training data
        #    (we will later use the sphering parameters obtained here to
        #     "sphere" the test data)
        print "sphering data..."
        v_sub = fvectors.mean(axis=0)
        fvectors -= v_sub
        v_div = fvectors.std(axis=0)
        scipy.putmask(v_div, v_div==0, 1)
        fvectors /= v_div
        return fvectors, v_sub, v_div
        
  
    def _dimr(self,fvectors,pca_threshold):
    
        # -- Reduce dimensionality using a pca / eigen subspace projection
        nvectors, vsize = fvectors.shape        
        if nvectors < vsize:
            print "pca...", 
            print fvectors.shape, "=>", 
            U,S,V = v1s_math.fastsvd(fvectors)
            eigvectors = V.T
            i = tot = 0
            S **= 2.
            while (tot <= pca_threshold) and (i < S.size):
                tot += S[i]/S.sum()
                i += 1
            eigvectors = eigvectors[:, :i+1]
            fvectors = scipy.dot(fvectors, eigvectors)
       
        return fvectors, eigvectors
       
   
    def _get_data(self,fvectors,flabels):
        print "creating dataset..."
        if PyML.__version__ < "0.7.0":
            data = PyML.datafunc.VectorDataSet(fvectors.astype(scipy.double),
                                                L=flabels)
        else:
            data = PyML.containers.VectorDataSet(fvectors.astype(scipy.double),
                                                  L=flabels)     
        return data
        
    def _get_sparse_data(self,fvectors,flabels):
        print "creating dataset..."
        if PyML.__version__ < "0.7.0":
            data = PyML.datafunc.SparseDataSet(fvectors.astype(scipy.double),
                                                L=flabels)
        else:
            data = PyML.containers.SparseDataSet(fvectors.astype(scipy.double),
                                                  L=flabels)     
        return data
        
        
    def _train_svm(self,train_fvectors,train_flabels,classlist_dict):     
        train_data = self._get_data(train_fvectors,train_flabels)

        print "creating classifier..."
        if len(classlist_dict) > 2:
            clas = multi.OneAgainstRest(svm.SVM())
        else:
            clas = PyML.svm.SVM()
        print "training svm..."
        clas.train(train_data,saveSpace=False)
        
        return clas, train_data
   
   
    def _safety_check(self,test_fnames,train_fnames):
        # a safety check that there are no duplicates across the test and 
        # train sets
        for test in test_fnames:
            if test in train_fnames:
                raise ValueError, "image already included in train set"



    # -------------------------------------------------------------------------
    def _get_fvectors_flabels_fnames(self, filelists_dict, model, idx_start,
                                     idx_end, hook=None):
        """ Read in data from a specific set of image files, apply the model,
            and generate feature vectors for each image.

        Inputs:
          filelists_dict -- dict of image files where:
                           . each key is a category,
                           . each corresponding value is a list of images
          model -- list with model parameters (cf. self.get_performance)
          idx_start -- start index in image list
          idx_end -- end index in image list
          hook -- hook function, if not None this function will take a vector
                  perform needed computation and return the resulting vector

        Outputs:
          fvectors -- array with feature vectors          
          flabels -- list with corresponding class labels
          fnames -- list with corresponding file names        

        """
        
        flabels = []
        fnames = []
        catlist = filelists_dict.keys()
        catlist.sort()

        # -- get fvectors size and initialize it
        print "Initializing..."
        f = filelists_dict[catlist[0]][0]
        if f in self._cache:
            vector = self._cache[f]
        else:
            vector = self._generate_repr(f, model)
            self._cache[f] = vector
            
        if hook is not None:
            vector = hook(vector)        
        nvectors = 0
        for cat in catlist:
            nvectors += len(filelists_dict[cat][idx_start:idx_end])

        fvectors = scipy.empty((nvectors, vector.size), 'f')

        # -- get all vectors
        i = 0
        for cat in catlist:
            print "get data from class:", cat

            for f in filelists_dict[cat][idx_start:idx_end]:
                txt = os.path.split(f)[-1]
                print "  file:", txt,
                
                if f in self._cache:
                    vector = self._cache[f]
                else:
                    vector = self._generate_repr(f, model)
                    self._cache[f] = vector
                    
                if hook is not None:
                    vector = hook(vector)        

                print "vsize:", vector.size

                fvectors[i] = vector
                flabels += [os.path.split(cat)[-1]]
                fnames += [f]
                i += 1
                
        return fvectors, flabels, fnames

    # -------------------------------------------------------------------------
    def _generate_repr(self, img_fname, model):
        """ Apply the simple V1-like model and rearrange the outputs into
            a feature vector suitable for further processing (e.g. 
            dimensionality reduction & classification). Model parameters
            determine both how the V1-like representation is built and which
            additional synthetic features (if any) are included in the 
            final feature vector.
            
            Most of the work done by this function is handled by the helper
            function _part_generate_repr.

        Inputs:
          img_fname -- image filename
          model -- list with model parameters (cf. self.get_performance)

        Outputs:
          fvector -- corresponding feature vector
          
        """

        all = []
        for params, featsel in model:
            r = self._part_generate_repr(img_fname, params, featsel)
            all += [r]
            
        fvector = scipy.concatenate(all)

        return fvector

    # -------------------------------------------------------------------------
    def _part_generate_repr(self, img_fname, params, featsel):
        """ Applies a simple V1-like model and generates a feature vector from
        its outputs. See description of _generate_repr, above.

        Inputs:
          img_fname -- image filename
          params -- representation parameters (dict)
          featsel -- features to include to the vector (dict)

        Outputs:
          fvector -- corresponding feature vector                  

        """
        
        # -- get image as an array
        orig_imga = v1s_funcs.get_image(img_fname, params['preproc']['max_edge'])

        # -- 0. preprocessing
        imga0 = orig_imga.astype('f') / 255.0        
        if imga0.ndim == 3:
            # grayscale conversion
            imga0 = 0.2989*imga0[:,:,0] + \
                    0.5870*imga0[:,:,1] + \
                    0.1140*imga0[:,:,2]
            
        # smoothing
        lsum_ksize = params['preproc']['lsum_ksize']
        mode = 'same'
        if lsum_ksize is not None:
            k = scipy.ones((lsum_ksize), 'f') / lsum_ksize
            imga0 = conv(conv(imga0, k[newaxis,:], mode), k[:,newaxis], mode)
            imga0 -= imga0.mean()
            if imga0.std() != 0:
                imga0 /= imga0.std()
        
        # -- 1. input normalization
        imga1 = v1s_funcs.v1s_norm(imga0[:,:,newaxis], **params['normin'])
        
        # -- 2. linear filtering
        filt_l = self._get_gabor_filters(params['filter'])
        imga2 = v1s_funcs.v1s_filter(imga1[:,:,0], filt_l)

        # -- 3. simple non-linear activation (clamping)
        minout = params['activ']['minout'] # sustain activity
        maxout = params['activ']['maxout'] # saturation
        imga3 = imga2.clip(minout, maxout)

        # -- 4. output normalization
        imga4 = v1s_funcs.v1s_norm(imga3, **params['normout'])

        # -- 5. volume dimension reduction
        imga5 = v1s_funcs.v1s_dimr(imga4, **params['dimr'])
        output = imga5
        
        # -- 6. handle features to include
        feat_l = []
        
        # include representation output ?
        f_output = featsel['output']
        if f_output:
            feat_l += [output.ravel()]

        # include grayscale values ?
        f_input_gray = featsel['input_gray']
        if f_input_gray is not None:
            shape = f_input_gray
            feat_l += [scipy.misc.imresize(imga0, shape).ravel()]
        
        # include color histograms ?
        f_input_colorhists = featsel['input_colorhists']
        if f_input_colorhists is not None:
            nbins = f_input_colorhists
            colorhists = empty((3,nbins), 'f')
            if orig_imga.ndim == 3:
                for d in xrange(3):
                    h = scipy.histogram(orig_imga[:,:,d].ravel(),
                                  bins=nbins,
                                  range=[0,255])
                    binvals = h[0].astype('f')
                    colorhists[d] = binvals
            else:
                h = scipy.histogram(orig_imga[:,:].ravel(),
                              bins=nbins,
                              range=[0,255])
                binvals = h[0].astype('f')
                colorhists[:] = binvals
                
            feat_l += [colorhists.ravel()]
        
        # include input norm histograms ? 
        f_normin_hists = featsel['normin_hists']
        if f_normin_hists is not None:
            division, nfeatures = f_normin_hists
            feat_l += [v1s_funcs.rephists(imga1, division, nfeatures)]
        
        # include filter output histograms ? 
        f_filter_hists = featsel['filter_hists']
        if f_filter_hists is not None:
            division, nfeatures = f_filter_hists
            feat_l += [v1s_funcs.rephists(imga2, division, nfeatures)]
        
        # include activation output histograms ?     
        f_activ_hists = featsel['activ_hists']
        if f_activ_hists is not None:
            division, nfeatures = f_activ_hists
            feat_l += [v1s_funcs.rephists(imga3, division, nfeatures)]
        
        # include output norm histograms ?     
        f_normout_hists = featsel['normout_hists']
        if f_normout_hists is not None:
            division, nfeatures = f_normout_hists
            feat_l += [v1s_funcs.rephists(imga4, division, nfeatures)]
        
        # include representation output histograms ? 
        f_dimr_hists = featsel['dimr_hists']
        if f_dimr_hists is not None:
            division, nfeatures = f_dimr_hists
            feat_l += [v1s_funcs.rephists(imga5, division, nfeatures)]

        # -- done !
        fvector = scipy.concatenate(feat_l)
        return fvector

    # -------------------------------------------------------------------------
    def _get_gabor_filters(self, params):
        """ Return a Gabor filterbank (generate it if needed)
            
        Inputs:
          params -- filters parameters (dict)

        Outputs:
          filt_l -- filterbank (list)

        """

        if self._filt_l is None:        
            # -- get parameters
            fh, fw = params['kshape']
            orients = params['orients']
            freqs = params['freqs']
            phases = params['phases']
            nf =  len(orients) * len(freqs) * len(phases)
            fbshape = nf, fh, fw
            gsw = fw/5.
            gsh = fw/5.
            xc = fw/2
            yc = fh/2
            filt_l = []

            # -- build the filterbank
            for freq in freqs:
                for orient in orients:
                    for phase in phases:
                        # create 2d gabor
                        filt = v1s_math.gabor2d(gsw,gsh,
                                       xc,yc,
                                       freq,orient,phase,
                                       (fw,fh))

                        # vectors for separable convolution
                        U,S,V = v1s_math.fastsvd(filt)
                        tot = 0
                        vectors = []
                        idx = 0
                        S **= 2.
                        while tot <= params['sep_threshold']:
                            row = (U[:,idx]*S[idx])[:, newaxis]
                            col = (V[idx,:])[newaxis, :]
                            vectors += [(row,col)]
                            tot += S[idx]/S.sum()
                            idx += 1                             

                        filt_l += [vectors]

            self._filt_l = filt_l
            
        return self._filt_l
       
       

#=-=-=-=-=-=-=-=-=-=Sequential code=-=-=-=-=-=-=-=-=-=-=-=-=-=       
       
@activate(lambda x : (x[0],x[1]), lambda x : x[2])       
def shuffle_images(param_fname,img_path,result_file):
    
    V = V1S(param_fname, img_path)
    
    filelists_dict = V._shuffle_images()
    
    F = open(result_file,'w')
    pickle.dump(filelists_dict,F)
    F.close()
    

@activate(lambda x : (x[0],x[1],x[2]), lambda x : x[3])      
def get_training_examples(param_fname,img_path, filelist_dict_file, result_file):

    V = V1S(param_fname, img_path)
 
    filelists_dict = pickle.load(open(filelist_dict_file))
    
    param_path = os.path.abspath(param_fname)
    v1s_params = {}
    execfile(param_path, {}, v1s_params)

    model = v1s_params['model']
    ntrain = V.ntrain
    train_fvectors, train_flabels, train_fnames = \
                         V._get_fvectors_flabels_fnames(filelists_dict,
                                                          model, 0, ntrain)   
 
 
    result_dict = {'train_fvectors' : train_fvectors, 'train_flabels' : train_flabels, 'train_fnames' : train_fnames}
    
    F = open(result_file,'w')
    pickle.dump(result_dict,F)
    F.close()
    
    
@activate(lambda x : (x[0],x[1],x[2]), lambda x : x[3]) 
def sphere(param_fname,img_path, train_examples_file,result_file):
    
    V = V1S(param_fname, img_path)
 
    training_examples = pickle.load(open(train_examples_file))
    train_fvectors = training_examples['train_fvectors']
        
    #sphere the data
    train_fvectors, v_sub, v_div = V._sphere(train_fvectors)
 
    result_dict = {'train_fvectors' : train_fvectors, 'v_sub' : v_sub, 'v_div' : v_div}
    
    F = open(result_file,'w')
    pickle.dump(result_dict,F)
    F.close()
     
     
@activate(lambda x : (x[0],x[1],x[2]), lambda x : x[3])     
def dimr(param_fname,img_path, sphered_results_file, result_file):      
        #reduce dimensions
 
    V = V1S(param_fname, img_path)
 
    sphered_results = pickle.load(open(sphered_results_file))
    train_fvectors = sphered_results['train_fvectors']
   
    param_path = os.path.abspath(param_fname)
    v1s_params = {}
    execfile(param_path, {}, v1s_params)
    
    pca_threshold = v1s_params['pca_threshold']
        
    train_fvectors, eigvectors = V._dimr(train_fvectors,pca_threshold)
 
    result_dict = {'train_fvectors' : train_fvectors, 'eigvectors' : eigvectors}    
    F = open(result_file,'w')
    pickle.dump(result_dict,F)
    F.close()    
        

@activate(lambda x : (x[0],x[1],x[2],x[3],x[4]), lambda x : x[5]) 
def train_svm(param_fname,img_path, filelists_dict_file, dimr_results_file, train_examples_file, classifier_file):        

    V = V1S(param_fname, img_path)
    
    filelists_dict = pickle.load(open(filelists_dict_file))
 
    dimr_results = pickle.load(open(dimr_results_file))
    train_fvectors = dimr_results['train_fvectors']
    
    train_examples = pickle.load(open(train_examples_file))
    train_flabels = train_examples['train_flabels']

    classifier, train_data = V._train_svm(train_fvectors,train_flabels,filelists_dict) 
        
    classifier.save(classifier_file)
    

     
@activate(lambda x : (x[0],x[1],x[2],x[3],x[4],x[5]), lambda x : x[6])     
def get_testing_examples(param_fname,img_path, filelist_dict_file, train_examples_file, sphered_results_file, dimr_results_file, result_file): 

    V = V1S(param_fname, img_path)
 
    filelists_dict = pickle.load(open(filelist_dict_file))
    
    param_path = os.path.abspath(param_fname)
    v1s_params = {}
    execfile(param_path, {}, v1s_params)

    model = v1s_params['model']
    ntest = V.ntest
    ntrain = V.ntrain

    dimr_results = pickle.load(open(dimr_results_file))
    train_fvectors = dimr_results['train_fvectors']
    eigvectors = dimr_results['eigvectors']
    
    sphered_results = pickle.load(open(sphered_results_file))
    v_sub = sphered_results['v_sub']
    v_div = sphered_results['v_div']
   
    train_examples = pickle.load(open(train_examples_file))
    train_fnames = train_examples['train_fnames']   
    
    #get hook
    hook_params = {'v_sub': v_sub, 'v_div' : v_div, 'eigvectors' : eigvectors}
    hook = V._get_hook(train_fvectors,**hook_params) 
     
    # -- Get testing examples  
    test_fvectors, test_flabels, test_fnames = \
                   V._get_fvectors_flabels_fnames(filelists_dict, model,
                                                     ntrain, ntrain+ntest,
                                                     hook)
    
    #safety check
    V._safety_check(test_fnames,train_fnames)
    
    
    result_dict = {'test_fvectors' : test_fvectors, 'test_flabels' : test_flabels, 'test_fnames' : test_fnames}
    
    F = open(result_file,'w')
    pickle.dump(result_dict,F)
    F.close()   
    

@activate(lambda x : (x[0],x[1],x[2],x[3]), lambda x : x[4])     
def test_svm(param_fname,img_path,test_examples_file,classifier_file,result_file):

    V = V1S(param_fname, img_path)
 
    test_examples = pickle.load(open(test_examples_file))
    test_fvectors = test_examples['test_fvectors']
    test_flabels = test_examples['test_flabels']

    classifier = PyML.classifiers.svm.loadSVM(classifier_file,labelsColumn=1)
    
    results = V._test_svm(test_fvectors,test_flabels,classifier)
 
    F = open(result_file,'w')
    pickle.dump(results,F)
    F.close()   
    
    
    
def trial_protocol(param_fname,img_path,result_dir,prefix = '',make_container = True):

    if prefix != '':
        prefix += '_'

    filelist_dict_file = os.path.join(result_dir,prefix + 'filelists_dict.pickle')
    train_examples_file = os.path.join(result_dir,prefix + 'train_examples.pickle')
    classifier_file = os.path.join(result_dir,prefix + 'classifier.pickle')
    sphered_results_file = os.path.join(result_dir,prefix + 'sphered_results.pickle')
    dimr_results_file = os.path.join(result_dir,prefix + 'dimr_results.pickle')
    test_examples_file = os.path.join(result_dir,prefix + 'test_examples.pickle')
    final_results_file = os.path.join(result_dir,prefix + 'final_results.pickle')


    D = []
    
    if make_container:
        D += [(prefix + 'initialize',MakeDir,(result_dir,))]
    
    D += [(prefix + 'shuffle_images',shuffle_images,(param_fname,img_path,filelist_dict_file)),
          (prefix + 'get_training_examples',get_training_examples,(param_fname,img_path, filelist_dict_file, train_examples_file)),
          (prefix + 'sphere', sphere, (param_fname,img_path, train_examples_file,sphered_results_file)),
          (prefix + 'dimr', dimr, (param_fname,img_path, sphered_results_file, dimr_results_file)),
          (prefix + 'train_svm', train_svm, (param_fname,img_path, filelist_dict_file, dimr_results_file, train_examples_file, classifier_file)),
          (prefix + 'get_testing_examples', get_testing_examples, (param_fname,img_path, filelist_dict_file, train_examples_file, sphered_results_file, dimr_results_file, test_examples_file)),
          (prefix + 'test_svm', test_svm, (param_fname,img_path,test_examples_file,classifier_file,final_results_file))]
    
    return D    
    
