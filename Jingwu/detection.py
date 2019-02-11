from __future__ import division

from functions import fn_fastSmooth

import os
import numpy as np

def dt_HR(params, hdr, filteredData):

    # Tyack & Clark 2000 cite Au (1993) in Hearing by Whales & Dolphins, Au
    # (ed.) stating that dolphins can distinguish clicks separated by as
    # little as 205 us.

    minGapSamples = np.ceil(params.mergeThr*hdr.fs/10**6)
    energy = filteredData ** 2
    candidatesRel = np.nonzero(energy>((params.countThresh**2)))[0]

    if len(candidatesRel):
        if params.saveNoise:
            noise = dt_getNoise(candidatesRel,len(energy), params, hdr)
            
        sStarts, sStops = dt_getDurations(candidatesRel, minGapSamples,len(energy))
        
        cStarts,cStops = dt_HR_expandRegion(params,hdr,sStarts,sStops,energy)
        
        completeClicks = np.array([cStarts,cStops]).T
    return completeClicks, noise

def dt_LR(energy, hdr, buffSamples, startK, stopK, params):
    # Find all events exceeding threshold.
    # Return times of those events
    ######################################################################

    # Flag times when the amplitude rises above a threshold
    aboveThreshold = np.nonzero(energy>((params.countThresh**2)))[0]
    if len(aboveThreshold) == 0:
        detectionsSec = []
        detectionsSample = []
    else:
        # sampleStart = floor(startK * hdr.fs)+1
        # sampleStop = floor(stopK * hdr.fs)+1

        # add a buffer on either side of detections.
        detStartSample = np.maximum(aboveThreshold - buffSamples, 0)
        detStopSample = np.minimum(aboveThreshold + buffSamples, len(energy)-1)

        detStart = np.maximum(((aboveThreshold - buffSamples)/hdr.fs) + startK, startK)
        detStop = np.minimum(((aboveThreshold + buffSamples)/hdr.fs) + startK, stopK)

        # Merge flags that are close together.
        if len(detStart) > 1:
            startsM, stopsM = dt_mergeCandidates(buffSamples/hdr.fs, detStop, detStart)
            startsSampleM, stopsSampleM = dt_mergeCandidates(buffSamples, detStopSample, detStartSample)
        else:
            startsM = detStart
            stopsM = detStop
            startsSampleM = detStartSample
            stopsSampleM = detStopSample

        detectionsSec = np.array([startsM, stopsM]).T
        detectionsSample = np.array([startsSampleM, stopsSampleM]).T
    return detectionsSample, detectionsSec


def dt_buildDirs(param):
    # build output directories
    try:
        # use outDir if specified
        param.metaDir = os.path.join(param.outDir, param.depl+'_'+'metadata')
    except AttributeError:
        param.metaDir = os.path.join(param.baseDir, param.depl+'_'+'metadata')
    
    if not os.path.exists(param.metaDir):
        os.makedirs(param.metaDir)


def dt_init_cParams(params):
    # Initialize vectors for main detector loop

    cParams = type('cParams',(object,),{})()
    n = 10**5
    cParams.clickTimes    = np.zeros((n,2))
    cParams.ppSignalVec   = np.zeros(n)
    cParams.durClickVec   = np.zeros(n)
    cParams.bw3dbVec      = np.zeros((n,3))
    cParams.specClickTfVec= np.zeros((n,len(params.specRange)))
    cParams.peakFrVec     = np.zeros(n)
    cParams.deltaEnvVec   = np.zeros(n)
    cParams.nDurVec       = np.zeros(n)
    # time series stored in cell arrays because length varies
    cParams.yFiltVec = np.zeros(n)
    cParams.yFiltBuffVec = np.zeros(n)

    if params.saveNoise:
        cParams.yNFiltVec = []
        cParams.specNoiseTfVec = []
    return cParams


def dt_chooseSegmentsRaw(hdr):
    dnum2sec = 60*60*24
    starts = np.array(hdr.raw.dnumStart)
    stops = np.array(hdr.raw.dnumEnd)
    startsSec = (starts - starts[0])*dnum2sec
    stopsSec = (stops - starts[0])*dnum2sec
    return startsSec, stopsSec


def dt_mergeCandidates(mergeThr, stops, starts):
    # merge candidates that are too close together so they will be considered
    # to be one larger detection
    c_startsM = []
    c_stopsM = []
    mergeL = 0
    while mergeL < len(starts):
        mergeR = mergeL+1
        c_startsM.append(starts[mergeL])
        while mergeR<len(starts) and starts[mergeR]-stops[mergeR-1] <= mergeThr:
            mergeR += 1
        c_stopsM.append(stops[mergeR-1])
        mergeL = mergeR
    return np.array(c_startsM), np.array(c_stopsM)


def dt_HR_expandRegion(params, hdr, sStarts, sStops, energy):
    # Expand region to lower thresholds

    N = len(energy)-1
    c_starts = np.zeros(len(sStarts))   # init complete clicks to single/partial clicks
    c_stops = np.zeros(len(sStarts))
    k=1

    dataSmooth = fn_fastSmooth(energy,15)
    thresh = np.percentile(energy,params.energyPrctile,interpolation='nearest')
    # [FUTURE] Parallelize here?
    for itr in range(len(sStarts)):
        rangeVec = np.arange(sStarts[itr],sStops[itr]+1)
        m = np.max(energy[rangeVec])

        largePeakList = np.sort(np.nonzero(energy[rangeVec] > .5*m))[0]
        midx = rangeVec[largePeakList[0]]
        
        # [FUTURE] TIME CONSUMING COMPUTATION
        leftmost = 4
        leftIdx = max(midx - 1,leftmost)
        while (leftIdx > leftmost) and np.mean(dataSmooth[leftIdx-4:leftIdx+1] > thresh) !=0: # /2
            leftIdx = leftIdx - 1

        rightmost = N-5
        rightIdx = midx+1
        while rightIdx < rightmost and np.mean(dataSmooth[rightIdx:rightIdx+5] > thresh) !=0: #+bpStd/2
            rightIdx = rightIdx+1
        c_starts[itr] = leftIdx
        c_stops[itr] = rightIdx

    if len(c_starts) > 1:
        idxs = np.argsort(c_starts)
        c_starts = c_starts[idxs]
        c_stops = c_stops[idxs]
        c_starts, c_stops = dt_mergeCandidates(params.mergeThr,c_stops,c_starts);
        # clf;plot(bpDataHi);hold on;plot(dataSmooth,'r');plot([c_starts,c_stops],zeros(size([c_starts,c_stops])),'*g');title(num2str(c_stops - c_starts));

    throwIdx = np.zeros(len(c_stops))
    for k2 in range(len(c_stops)):
        # Discard short signals or those that run past end of signal
        if c_stops[k2] >= N-2 or\
            c_stops[k2]-c_starts[k2] < params.clickSampleLims[0] or\
            c_stops[k2]-c_starts[k2] > params.clickSampleLims[1]:
            throwIdx[k2] = 1

    # [TODO] Hanlde illegal cases
    c_starts[throwIdx==1] = 0
    c_stops[throwIdx==1] = 0
    # [WARN] 0 remained in the output array
    c_starts, c_stops
    return c_starts, c_stops


def dt_getNoise(candidatesRel, dataLen, params, hdr):
    # Get noise

    maxClickSamples = params.clickSampleLims[1]
    candidatesRelwEnds = np.concatenate([[0], candidatesRel,[dataLen-1]])
    dCR = np.diff(candidatesRelwEnds)
    mI = np.argmax(dCR)
    # look for a stretch of data that has no detections
    if dataLen - (candidatesRelwEnds[mI]+maxClickSamples/2) > maxClickSamples:
        noiseStart = candidatesRelwEnds[mI]+maxClickSamples/2
        noiseTimes = [noiseStart, noiseStart+maxClickSamples]
    return noiseTimes


def dt_getDurations(detIndices, mergeThreshold, idxMax):
    # [start, duration] = spDurations(Indices)
    # Given a vector of indices into another vector, determine
    # the starting point of each distinct region and how long it is.
    
    # [WARN] Careful about the index here

    if not len(detIndices):
        stop = []
        start = []
    #     return
    # find skips in sequence
    diffs = np.diff(detIndices)

    # 1st index is always a start 
    # last index is always a stop
    # indices whose first difference is greater than one denote a 
    # start or stop boundary.
    startPositions = np.concatenate([[0], np.nonzero(diffs>mergeThreshold)[0]+1])
    start = detIndices[startPositions]
    if len(startPositions) > 0:
        # [QUESTION] -1 or not
        stopPositions = np.concatenate([startPositions[1:]-1, [len(detIndices)-1]])
        stop = detIndices[stopPositions]
    else:
        stop = np.min(detIndices[startPositions]+1,idxMax)
    return start, stop