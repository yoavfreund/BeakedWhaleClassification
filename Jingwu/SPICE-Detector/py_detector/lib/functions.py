"""
This file contains functions that support detection process
"""
import os
import math
import numpy as np
import pickle
from scipy import signal
from scipy.interpolate import interp1d

def fn_findXWAVs(directory, ext):
    """ 
    Performs a recursive file search
    returns a list of file names ending with extension
    
    Args:
        directory (string): base directory
        ext (string): specific extension of returning files 
    
    Returns:
        fList (list) : a list of file names ending with extension 
    """
    
    # [TODO] how to read files when paralleling 
    # [NOT DONE] check for duplicate file names and warn user if found
    
    fList = list()
    for root, dirs, files in os.walk(directory):
        for xfile in files:
            if xfile.lower().endswith('.'+ext):
                fList.append(os.path.join(root, xfile))
    return fList

def fn_buildFilters(params, fs):
    """ 
    Build signal filter for bandpassing raw data according to the user settings and sample rate
    if sampling rate is same as previous, no need to build new filter
    
    Args:
        params (type): AnonymousClass of setting values
        fs (int): sample rate (frequency)
    
    Returns:
        previousFs (int): previous sample rate (frequency)
        params (type): AnonymousClass of setting values with filter added
    """
    bandPassRange = params.bpRanges
    params.filtType = 'bandpass'
    params.filterSignal = True
    
    # Handle different filter cases:
    # 1) low pass
    if params.bpRanges[0] == 0:
        # they only specified a top freqency cutoff, so we need a low pass
        # filter
        bandPassRange = params.bpRanges[1]
        params.filtType = 'low'
        if bandpassRange == fs/2:
            # they didn't specify any cutoffs, so we need no filter
            params.filterSignal = False
            
    # 2) High passs
    if params.bpRanges[1] == fs/2 and params.filterSignal:
        # they only specified a lower freqency cutoff, so we need a high pass
        # filter
        bandPassRange = params.bpRanges[0]
        params.filtType = 'high'
    
    if params.filterSignal:
        params.fB, params.fA = signal.butter(params.filterOrder, bandPassRange/(fs/2),btype=params.filtType)
        
    # filtTaps = length(fB)
    previousFs = fs
    
    params.fftSize = int(math.ceil(fs * params.frameLengthUs / 10**6))
    if params.fftSize % 2 == 1:
        params.fftSize = params.fftSize - 1  # Avoid odd length of fft

    params.fftWindow = signal.windows.hann(params.fftSize)

    lowSpecIdx = int(params.bpRanges[0]/fs*params.fftSize)
    highSpecIdx = int(params.bpRanges[1]/fs*params.fftSize)

    params.specRange = np.arange(lowSpecIdx, highSpecIdx+1)
    params.binWidth_Hz = fs / params.fftSize
    params.binWidth_kHz = params.binWidth_Hz / 1000
    params.freq_kHz = params.specRange*params.binWidth_kHz  # calculate frequency axis
    return previousFs, params

def fn_getFileset(directory, fullFileNames):
    """ 
    Make list of what you're going to name your output files, for easy reference later.
    returns a list of same files name but ending up with '.txt' redirecting to a directory

    Args:
        directory (string): output directory
        fullFileNames (list): a list of file names
    
    Returns:
        fullLabels (list): a list of file names with '.txt' ending for storing outputs
    """
    fullLabels = [None] * len(fullFileNames)
    for i in range(len(fullFileNames)):
        xfile = fullFileNames[i]
        fname = os.path.basename(xfile)
        pureName = os.path.splitext(os.path.splitext(fname)[0])[0]
        fullLabels[i] = os.path.join(directory, pureName+'.txt')
    return fullLabels

def fn_interp_tf(params):
    """
    If a transfer function is provided, interpolate to desired frequency bins
    
    Args:
        params (type): AnonymousClass of setting values
    
    Returns:
        params (type): AnonymousClass of setting values with transfer function values added
    """

    # Determine the frequencies for which we need the transfer function
    params.xfr_f = np.arange((params.specRange[0]-1)*params.binWidth_Hz,\
                             (params.specRange[-1])*params.binWidth_Hz,\
                             params.binWidth_Hz)
    if params.tfFullFile:
        # [TODO] nargin > 1 in dtf_map
        params.xfrOffset = fn_tfMap(params.tfFullFile, params.xfr_f)
    else:
        # if you didn't provide a tf function, then just create a
        # vector of zeros of the right size.
        params.xfrOffset = np.zeros(len(params.xfr_f))
    return params

def fn_tfMap(tf_fname, f_desired=[]):
    """
    Given a path to a transfer function file open it and 
    interpollate to curve to match desired frequency vector.
    
    Args:
        tf_fname (string): transfer function file path
        f_desired (list, optional): desired frequency vector
    
    Returns:
        uppc: frequencies that match desired frequency vector
    
    Raises:
        Exception: Errors in transfer function
    """
    f, uppc = np.genfromtxt(tf_fname).T
    
    # [TODO] sum(f_desired ~= f)
    if len(f_desired) and len(f_desired)!=len(f) and sum(f_desired)!=sum(f):
        if len(set(f)) != len(f):
            raise Exception('Duplicate frequencies detected in transfer function.')
            return
        func = interp1d(f, uppc, 'linear')
        uppc = func(f_desired)
    return uppc

def fn_fastSmooth(Y, w, stype=1, ends=0):
    # fastbsmooth(Y,w,stype,ends) smooths vector Y with smooth 
    #  of width w. Version 2.0, May 2008.
    # The argument "stype" determines the smooth stype:
    #   If stype=1, rectangular (sliding-average or boxcar) 
    #   If stype=2, triangular (2 passes of sliding-average)
    #   If stype=3, pseudo-Gaussian (3 passes of sliding-average)
    # The argument "ends" controls how the "ends" of the signal 
    # (the first w/2 points and the last w/2 points) are handled.
    #   If ends=0, the ends are zero.  (In this mode the elapsed 
    #     time is independent of the smooth width). The fastest.
    #   If ends=1, the ends are smoothed with progressively 
    #     smaller smooths the closer to the end. (In this mode the  
    #     elapsed time increases with increasing smooth widths).
    # fastsmooth(Y,w,stype) smooths with ends=0.
    # fastsmooth(Y,w) smooths with stype=1 and ends=0.
    # Example:
    # fastsmooth([1 1 1 10 10 10 1 1 1 1],3)= [0 1 4 7 10 7 4 1 1 0]
    # fastsmooth([1 1 1 10 10 10 1 1 1 1],3,1,1)= [1 1 4 7 10 7 4 1 1 1]
    #  T. C. O'Haver, May, 2008.
    
    # [WARN] CARE ABOUT LOOP INDEX
    # [FUTURE] PARALLEL FOR LOOP?
    
    def sa(Y,smoothwidth,ends):
        w = int(np.rint(smoothwidth))
        SumPoints = sum(Y[:w])
        s = np.zeros(len(Y))
        halfw = int(np.rint(w/2))
        L = len(Y)
        for k in range(L-w):
            s[k+halfw-1]=SumPoints
            SumPoints=SumPoints-Y[k]
            SumPoints=SumPoints+Y[k+w]
            
        s[L-w+halfw-1]=np.sum(Y[L-w:])
        SmoothY = s/w
        
        if ends==1:
            startpoint = int((smoothwidth + 1)/2)
            SmoothY[0] = (Y[0]+Y[1])/2
            for k in range(1,startpoint):
                SmoothY[k] = np.mean(Y[1:2*k+1])
                SmoothY[L-k] = np.mean(Y[L-2*k:])
            SmoothY[-1]=(Y[-1]+Y[-2])/2
            
        return SmoothY
    
    if stype == 1:
        SmoothY=sa(Y,w,ends)
    elif stype == 2:
        SmoothY=sa(sa(Y,w,ends),w,ends)
    elif stype == 3:
        SmoothY=sa(sa(sa(Y,w,ends),w,ends),w,ends)
    return SmoothY

def fn_saveDets2pkl(fileName, cParams, f, hdr, params):
    """
    Save setting values to pickle file
    
    Args:
        fileName (string): output file that stores setting values
        cParams (type): AnonymousClass of current setting values 
        f (int): frequency
        hdr (type): AnonymousClass of header metadata
        params (type): AnonymousClass of loaded(default) setting values 
    """
    def anomyousClass_to_dict(anonymousClass):
        """
        Convert type to dictionary for pickling
        
        Args:
            anonymousClass (type): AnonymousClass of setting values 
        
        Returns:
            anonymousDict (dict): Dictionary of setting values in key-value pairs
        """
        # [TODO] a alternative way to dump anonymousClass
        # 1. better way to use pickle dump
        # 2. change the structure of storage, like dict
        anonymousDict = dict()
        for key, value in list(vars(anonymousClass).items()):
            if hasattr(value,'__dict__') or hasattr(value,'__slots__'):
                anonymousDict[key] = anomyousClass_to_dict(value)
            else:
                anonymousDict[key] = value
        return anonymousDict

    # Can't save directly in parfor loop, so save externally    
    clickTimes = cParams.clickTimes
    ppSignal = cParams.ppSignalVec
    yFiltBuff = cParams.yFiltBuffVec
    specClickTf = cParams.specClickTfVec
    yNFilt = cParams.yNFiltVec
    if params.saveForTPWS: # only save what you need to build a TPWS file
        with open(fileName, 'wb') as fw:
            pickle.dump([clickTimes, ppSignal, f, anomyousClass_to_dict(hdr),
                         specClickTf, yFiltBuff, anomyousClass_to_dict(params)], fw)
            if params.saveNoise:
                pickle.dump([yNFilt], fw)
    else:
        durClick = cParams.durClickVec
        nDur = cParams.nDurVec
        deltaEnv = cParams.deltaEnvVec
        bw3db = cParams.bw3dbVec
        yFilt = cParams.yFiltVec
        peakFr = cParams.peakFrVec
        with open(fileName, 'wb') as fw:
            pickle.dump([clickTimes, ppSignal, durClick, f, anomyousClass_to_dict(hdr), 
                         nDur, deltaEnv, bw3db, yFilt, specClickTf, peakFr,yFiltBuff, 
                         anomyousClass_to_dict(params)], fw)
            if params.saveNoise:
                yNFilt = cParams.yNFiltVec
                specNoiseTf = cParams.yNFiltVec
                pickle.dump([yNFilt, specNoiseTf], fw)