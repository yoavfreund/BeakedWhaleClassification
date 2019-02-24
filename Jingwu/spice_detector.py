from __future__ import division

import argparse
import yaml
import numpy as np
import time
import warnings

from scipy.signal import filtfilt
from lib.wavio import *
from lib.functions import *
from lib.detection import *

def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def dt_batch(fullFiles, fullLabels, detParams, encounterTimes, runMode):

    N = len(fullFiles)
    detParams.previousFs = 0 # make sure we build filters on first pass

    # get file type list
    # fTypes = io_getFileType(fullFiles);

    for idx1 in range(N): # for each data file
        params = detParams
        
        currentRecFile = fullFileNames[idx1]
        outFileName = fullLabels[idx1]
        print 'beginning file %d of %d \n' % (idx1, N)
        start = time.time()

        # read file header
        hdr = io_readXWavHeader(currentRecFile)

        # [TODO] extend options for wav files
        # divide xwav by raw file
        startsSec, stopsSec = dt_chooseSegmentsRaw(hdr)

        if hdr.fs != params.previousFs:
            # otherwise, if this is the first time through, build your filters,
            # only need to do this once though, so if you already have this
            # info, this step is skipped
            
            previousFs, _ = fn_buildFilters(params, hdr.fs)
            params.previousFs = previousFs;
            params = fn_interp_tf(params)
            if 'countThresh' not in vars(params) or params.countThresh == 0:
                params.countThresh = (10**((params.dBppThreshold - np.median(params.xfrOffset))/20))/2

        cParams = dt_init_cParams(params) # set up storage for HR output.
        sIdx = 0

        # Open audio file
        with open(currentRecFile, 'r') as fh:
            buffSamples = int(params.LRbuffer*hdr.fs)
            # Loop through search area, running short term detectors
            params.clickSampleLims = np.int64(np.ceil((hdr.fs/10**6)*\
                                                 np.array([params.delphClickDurLims[0]*.75,\
                                                 params.delphClickDurLims[1]*1.25])))
            for k in range(len(startsSec)):
                
                startK = startsSec[k]
                stopK = stopsSec[k]

                # [TODO] Read in data segment for WAV format
                dat = io_readRaw(fh, hdr, k, params.channel)
                
                # bandpass
                if params.filterSignal:
                    padlen = 3*(max(len(params.fB),len(params.fA))-1)
                    filtData = filtfilt(params.fB, params.fA, dat, padtype='odd', padlen=padlen)
                else:
                    filtData = dat
                
                energy = filtData ** 2
                
                ### Run LR detection to identify candidates
                ts = time.time()
                detectionsSample,detectionsSec =  dt_LR(energy,hdr,buffSamples,startK,stopK,params)
                
                # [2/9]
                # [FUTURE] PARALLEL in detectionsSample
                ts = time.time()
                for iD in range(len(detectionsSample)):
                    # [FUTURE] HOW TO EXTRACT SUBARRAY FROM NP ARRAY WITH DIFFERENT LENGTH
                    filtSegment = filtData[detectionsSample[iD][0]:detectionsSample[iD][1]+1]
                    clicks, noise = dt_HR(params, hdr, filtSegment)
                    if len(clicks):
                        # if we're in here, it's because we detected one or more possible
                        # clicks in the kth segment of data
                        # Make sure our click candidates aren't clipped
                        validClicks = dt_pruneClipping(clicks, params, hdr, filtSegment)

                        # Look at power spectrum of clicks, and remove those that don't
                        # meet peak frequency and bandwidth requirements
                        clicks = clicks[validClicks==1]
                        # Compute click parameters to decide if the detection should be kept
                        clickDets,f = dt_parameters(noise, filtSegment, params, clicks, hdr)
                        if len(clickDets.clickInd):
                            # populate cParams
                            cParams, sIdx = dt_populate_cParams(clicks, params, clickDets,
                                                                detectionsSec[iD][0], hdr, sIdx, cParams)
        
        print 'done with %s\n'%currentRecFile
        end = time.time()
        print round(end-start,2), ' s\n\n'

        cParams = dt_prune_cParams(cParams,sIdx)

        # Run post processing to remove rogue loner clicks, prior to writing
        # the remaining output files.
        clickTimes = cParams.clickTimes[cParams.clickTimes[:,0].argsort()]

        encounterTimes = []

        keepFlag = dt_postproc(outFileName,clickTimes,params,hdr,encounterTimes)
        keepIdx = np.where(keepFlag==1)[0]

        baseFileName = os.path.splitext(outFileName)[0]
        fn_saveDets2pkl(baseFileName+'.pkl',cParams,f,hdr,params)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='spice_detector')
    parser.add_argument('-s', '--setting', action='store', required=True, dest='detParamsFile', help='detector settings file', type=str)
    args = vars(parser.parse_args())

    runMode = 'batchRun' # default to batch. Need to implement guiRun.
    config = load_config(args['detParamsFile'])
    detParams = type('DetParams',(object,),config)()

    detParams = dt_buildDirs(detParams)

    # Build list of (x)wav names in the base directory.
    # Right now only wav and xwav files are looked for.
    fullFileNames = fn_findXWAVs(detParams.baseDir, 'x.wav')

    encounterTimes = list()

    # return a list of files to be built
    fullLabels = fn_getFileset(detParams.metaDir, fullFileNames)

    if len(fullFileNames):
        print 'Beginning detection\n\n'
        dt_batch(fullFileNames,fullLabels,detParams,encounterTimes,runMode)
    else:
        print 'Error: No wav/xwav files found'