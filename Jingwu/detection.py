from __future__ import division

from functions import fn_fastSmooth

import os
import numpy as np
from scipy.signal import hilbert

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

    cParams = type('AnonymousClass',(object,),{})()
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
    cParams.yFiltVec = [None] * n
    cParams.yFiltBuffVec = [None] * n

    if params.saveNoise:
        cParams.yNFiltVec = list()
        cParams.specNoiseTfVec = list()
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
    c_starts = np.zeros(len(sStarts), dtype=int)   # init complete clicks to single/partial clicks
    c_stops = np.zeros(len(sStarts), dtype=int)
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

    c_starts = np.delete(c_starts, np.where(throwIdx==1)[0])
    c_stops = np.delete(c_stops, np.where(throwIdx==1)[0])
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
    return np.array(noiseTimes).astype(int)


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

def dt_prune_cParams(cParams, sIdx):

    eIdx = sIdx
    # prune off any extra cells that weren't filled
    cParams.clickTimes = cParams.clickTimes[:eIdx]
    cParams.ppSignalVec = cParams.ppSignalVec[:eIdx]
    cParams.durClickVec = cParams.durClickVec[:eIdx]
    cParams.bw3dbVec = cParams.bw3dbVec[:eIdx]
    cParams.yFiltVec = cParams.yFiltVec[:eIdx]
    cParams.specClickTfVec = cParams.specClickTfVec[:eIdx]
    cParams.peakFrVec = cParams.peakFrVec[:eIdx]
    cParams.yFiltBuffVec = cParams.yFiltBuffVec[:eIdx]
    cParams.deltaEnvVec = cParams.deltaEnvVec[:eIdx]
    cParams.nDurVec = cParams.nDurVec[:eIdx]
    return cParams

def dt_processValidClicks(clicks, clickDets, startsK, hdr):
    # Write click times to .ctg label file

    clkStart = [None] * len(clickDets.clickInd)
    clkEnd = [None] * len(clickDets.clickInd)

    for c in range(len(clickDets.clickInd)):
        cI = clickDets.clickInd[c]
        currentClickStart = startsK + clicks[cI,0]/hdr.fs # start time
        currentClickEnd = startsK + clicks[cI,1]/hdr.fs

        # Compute parameters of click frames
        clkStart[c] = currentClickStart
        clkEnd[c] = currentClickEnd
    return np.array(clkStart), np.array(clkEnd)

def dt_populate_cParams(clicks, params, clickDets, starts, hdr, sIdx, cParams):
    clkStarts, clkEnds = dt_processValidClicks(clicks, clickDets, starts, hdr)
    eIdx = sIdx + len(clickDets.nDur)
    cParams.clickTimes[sIdx:eIdx,0] = clkStarts
    cParams.clickTimes[sIdx:eIdx,1] = clkEnds
    cParams.ppSignalVec[sIdx:eIdx] = clickDets.ppSignal
    cParams.durClickVec[sIdx:eIdx] = clickDets.durClick
    cParams.bw3dbVec[sIdx:eIdx] = clickDets.bw3db
    cParams.yFiltVec[sIdx:eIdx]= clickDets.yFilt
    cParams.specClickTfVec[sIdx:eIdx] = clickDets.specClickTf
    cParams.peakFrVec[sIdx:eIdx] = clickDets.peakFr
    cParams.yFiltBuffVec[sIdx:eIdx] = clickDets.yFiltBuff
    cParams.deltaEnvVec[sIdx:eIdx] = clickDets.deltaEnv
    cParams.nDurVec[sIdx:eIdx] = clickDets.nDur

    if params.saveNoise:
        if len(clickDets.yNFilt):
            cParams.yNFiltVec.append(clickDets.yNFilt)
            cParams.specNoiseTfVec.append(clickDets.specNoiseTf)

    sIdx = eIdx
    return cParams, sIdx

def dt_pruneClipping(clicks, params, hdr, filteredData):
    # Prune out detections that are too high amplitude, and therefore likely
    # clipped.

    validClicks = np.ones(len(clicks), dtype=int)  # assume all are good to begin
    for c in range(len(clicks)):
        # Go through the segments of interest one by one, and make sure they
        # don't exceed clip threshold.
        if any(abs(filteredData[clicks[c,0]:clicks[c,1]+1])) > params.clipThreshold *(2^hdr.nBits)/2:
            validClicks[c] = 0
    return validClicks

def dt_parameters(noiseIn, filteredData, params, clicks, hdr):
    #Take timeseries out of existing file, convert from normalized data to
    #counts
    #1) calculate spectral received levels RL for click and preceding noise:
    #calculate spectra, account for bin width to reach dB re counts^2/Hz,
    #add transfer function, compute peak frequency and bandwidth
    #2) calculate RLpp at peak frequency: find min & max value of timeseries,
    #convert to dB, add transfer function value of peak frequency (should come
    #out to be about 9dB lower than value of spectra at peak frequency)
    #3) Prune out clicks that don't fall in expected peak frequency, 3dB
    #bandwidth/duration range, or which are not high enough amplitude
    #(ppSignal)
    # ** There's code in here to compute a noise spectrum alongside the click
    # spectrum. It should at least get you started if you want that sort of
    # thing.

    ############################################
    # Initialize variables
    N = len(params.fftWindow)
    f = np.arange(0, ((hdr.fs/2)/1000)*(N/2+1)/(N/2), ((hdr.fs/2)/1000)/(N/2))
    f = f[params.specRange-1]
    sub = 10*np.log10(hdr.fs/N)

    ppSignal = np.zeros(len(clicks))
    durClick =  np.zeros(len(clicks))
    bw3db = np.zeros((len(clicks),3))
    yFilt = [None] * len(clicks)
    yFiltBuff = [None] * len(clicks)
    specClickTf = np.zeros((len(clicks), len(f)))
    peakFr = np.zeros(len(clicks))
    # cDLims = ceil([params.minClick_us, params.maxClick_us]./(hdr.fs/1e6))
    envDurLim = np.ceil(np.array(params.delphClickDurLims)*(hdr.fs/1e6))
    nDur = np.zeros(len(clicks))
    deltaEnv = np.zeros(len(clicks))

    if params.saveNoise:

        if len(noiseIn):
            yNFilt = filteredData[noiseIn[0]:noiseIn[1]+1]

            noiseWLen = len(yNFilt)
            noiseWin = np.hanning(noiseWLen)
            wNoise = np.zeros(N)
            wNoise[0:noiseWLen] = noiseWin*yNFilt
            spNoise = 20*np.log10(abs(np.fft.fft(wNoise,N)))
            spNoiseSub = spNoise-sub
            spNoiseSub = spNoiseSub[0:N//2]
            specNoiseTf = spNoiseSub[params.specRange-1]+params.xfrOffset
        else:
            yNFilt = np.array([])
            specNoiseTf = np.array([])

    # Add small buffer to edges of clicks
    buffVal = int(hdr.fs*params.HRbuffer)

    for c in range(len(clicks)):
        # Pull out band passed click timeseries
        # [TODO] INCONSISTENCY HERE
        yFiltBuff[c] = filteredData[max(clicks[c,0]-buffVal,0):min(clicks[c,1]+buffVal,len(filteredData)-1)+1]
        yFilt[c] = filteredData[clicks[c,0]:clicks[c,1]+1]

        click = yFilt[c]
        clickBuff = yFiltBuff[c]
        ###############################################################
        # Calculate duration in samples
        durClick[c] = clicks[c,1]-clicks[c,0]

        # Compute click spectrum
        winLength = min(len(clickBuff), N)
        wind = np.hanning(winLength)
        wClick = np.zeros(N)
        wClick[0:winLength] = clickBuff[0:winLength]*wind
        spClick = 20*np.log10(abs(np.fft.fft(wClick,N)))

        # account for bin width
        spClickSub = spClick-sub

        #reduce data to first half of spectra
        spClickSub = spClickSub[0:N//2]
        specClickTf[c] = spClickSub[params.specRange-1]+params.xfrOffset

        #####
        # calculate peak click frequency
        # max value in the first half samples of the spectrogram

        posMx = np.argmax(specClickTf[c])
        valMx = specClickTf[c][posMx]
        peakFr[c] = f[posMx] #peak frequency in kHz

        #################
        # calculate click envelope (code & concept from SBP 2014):
        # env = sqrt((real(pre_env)).^2+(imag(pre_env)).^2) #Au 1993, S.178, equation 9-4
        env = abs(hilbert(click))

        #calculate energy duration over x# energy
        env = env - min(env)
        env = env/max(env)

        #determine if the slope of the envelope is positive or negative
        #above x# energy
        aboveThr = np.where(env>=params.energyThr)[0]
        direction = np.zeros(len(aboveThr), dtype=int)

        for a in range(len(aboveThr)):
            if aboveThr[a]>0 and aboveThr[a]<len(env)-1:
                # if it's not the first or last element fo the envelope, then
                # -1 is for negative slope, +1 is for + slope
                delta = env[aboveThr[a]+1]-env[aboveThr[a]]
                if delta>=0:
                    direction[a] = 1
                else:
                    direction[a] = -1
            elif aboveThr[a] == 0:
                # if you're looking at the first element of the envelope
                # above the energy threshold, consider slope to be negative
                direction[a] = -1
            else:  
                # if you're looking at the last element of the envelope
                # above the energy threshold, consider slope to be positive
                direction[a] = 1

        # find the first value above threshold with positive slope and find
        # the last above with negative slope
        lowIdx = aboveThr[np.where(direction==1)[0][0]]
        negative = np.where(direction==-1)[0]

        if not len(negative):
            highIdx = aboveThr[-1]
        else:
            highIdx = aboveThr[negative[-1]]
        nDur[c] = highIdx - lowIdx + 1

        #compare maximum first half of points with second half.
        halves = int(np.ceil(nDur[c]/2))
        env1max = max(env[lowIdx:min([lowIdx+halves+1,len(env)-1])])
        env2max = max(env[min([lowIdx+halves+1,len(env)-1]):])
        deltaEnv[c] = env1max-env2max

        #########################################

        # calculate bandwidth
        # -3dB bandwidth
        # calculation of -3dB bandwidth - amplitude associated with the halfpower points of a pressure pulse (see Au 1993, params.118)
        low = valMx-3 # p1/2power = 10log(p^2max/2) = 20log(pmax)-3dB = 0.707*pmax 1/10^(3/20)=0.707
        #walk along spectrogram until low is reached on either side
        slopeup = np.fliplr(specClickTf[c:c+1,0:posMx+1])
        slopedown = specClickTf[c,posMx:len(specClickTf[c])+1]

        e3dB = np.argmax(slopeup<low)
        o3dB = np.argmax(slopedown<low)

        #calculation from spectrogram -> from 0 to 100kHz in 256 steps (FFT=512)
        # [TODO] ITS STRANGE TO USE INDEX FOR CALCULATION
        # high3dB = (hdr.fs/(2*1000))*((posMx+o3dB+2)/len(specClickTf[c])) #-3dB highest frequency in kHz
        high3dB = (hdr.fs/(2*1000))*((posMx+o3dB)/len(specClickTf[c])) #-3dB highest frequency in kHz
        low3dB = (hdr.fs/(2*1000))*((posMx-e3dB)/len(specClickTf[c])) #-3dB lowest frequency in kHz
        bw3 = high3dB-low3dB

        bw3db[c] = [low3dB, high3dB, bw3]

        #####
        #calculate RLpp at peak frequency: find min/max value of timeseries,
        #convert to dB, add transfer function value of peak frequency (should come
        #out to be about 9dB lower than value of spectra at peak frequency)

        # find lowest and highest number in timeseries (counts) and add those
        high = max(click)
        low = min(click)
        ppCount = high+abs(low)

        #calculate dB value of counts and add transfer function value at peak
        #frequency to get ppSignal (dB re 1uPa)
        P = 20*np.log10(ppCount)

        peakLow=np.floor(peakFr[c])
        fLow=np.where(f>=peakLow)[0]

        #add PtfN transfer function at peak frequency to P
        tfPeak = params.xfrOffset[fLow[0]]
        ppSignal[c] = P+tfPeak

    validClicks = np.ones(len(ppSignal))

    # Check parameter values for each click
    for idx in range(len(ppSignal)):
        tfVec = [deltaEnv[idx] < params.dEvLims[0], peakFr[idx] < params.cutPeakBelowKHz,
                 peakFr[idx] > params.cutPeakAboveKHz, nDur[idx] > envDurLim[1],
                 nDur[idx] < envDurLim[0], durClick[idx] < params.delphClickDurLims[0],
                 durClick[idx] > params.delphClickDurLims[1]]
        if ppSignal[idx] < params.dBppThreshold:
            validClicks[idx] = 0
        elif sum(tfVec)>0:
            validClicks[idx] = 0


    clickInd = np.where(validClicks == 1)[0]
    clickDets = type('AnonymousClass',(object,),{})()
    clickDets.clickInd = clickInd
    # throw out clicks that don't fall in desired ranges
    clickDets.ppSignal = ppSignal[clickInd]
    clickDets.durClick =  durClick[clickInd]
    clickDets.bw3db = bw3db[clickInd]
    # frames = frames{clickInd}
    clickDets.yFilt = np.array(yFilt)[clickInd]
    clickDets.yFiltBuff = np.array(yFiltBuff)[clickInd]
    clickDets.specClickTf = specClickTf[clickInd]
    clickDets.peakFr = peakFr[clickInd]
    clickDets.deltaEnv = deltaEnv[clickInd]
    clickDets.nDur = nDur[clickInd]

    if params.saveNoise:
        clickDets.specNoiseTf = specNoiseTf
        clickDets.yNFilt = yNFilt
    return clickDets, f


def dt_postproc(outFileName, clickTimes, params, hdr, encounterTimes):

    # Step through vector of click times, looking forward and back to throw out
    # solo clicks, and pairs of clicks, if they are too far away from a cluster
    # of clicks with >2 members.
    # outputs a vector of pruned times, and a vector flagging which members
    # should be removed from other variables.
    # Writes pruned times to .pTg file.

    delFlag = np.ones(len(clickTimes), dtype=int) # t/f vector of click deletion flags. 
    # starts as all 1 to keep all clicks. Elements switch to zero as clicks are
    # flagged for deletion.

    ### Get rid of lone clicks ###
    if params.rmLonerClicks:
       
        # Step through deleting clicks that are too far from their preceeding
        # and following click    
        if len(clickTimes) > 2:
            for itr1 in range(len(clickTimes)):
                if itr1 == 0:
                    if clickTimes[itr1+1,1]-clickTimes[itr1,0]>params.maxNeighbor:
                        delFlag[itr1]= 0
                elif itr1 == len(clickTimes)-1:
                    prevIdxs = np.where[delFlag[0:itr1]==1][0]
                    if not len(prevIdxs):
                        delFlag[itr1] = 0
                    prevClick = max(prevIdxs)
                    if clickTimes[itr1,1] - clickTimes[prevClick,0]>params.maxNeighbor:
                        delFlag[itr1] = 0
                else:
                    prevIdxs = np.where[delFlag[0:itr1]==1][0]
                    if not len(prevIdxs):
                        if clickTimes[itr1+1,1] - clickTimes[itr1,0]>params.maxNeighbor:
                            delFlag[itr1] = 0
                    prevClick = max(prevIdxs)
                    if clickTimes[itr1,1] - clickTimes[prevClick,0]>params.maxNeighbor\
                        and clickTimes[itr1+1,1]-clickTimes[itr1,0]>params.maxNeighbor:
                        delFlag[itr1] = 0
        else:
            delFlag = np.zeros(len(clickTimes))
    # [TODO]: Get rid of pulsed calls

    # get rid of duplicate times:
    if len(clickTimes) > 1:
        dtimes = np.diff(clickTimes[:,0])
        closeStarts = np.where(dtimes<0.00002)[0]
        delFlag[closeStarts+1] = 0

    if params.rmEchos:
        # Added 150318 KPM - remove echoes from captive recordings.  Lock out
        # period N seconds from first click detection of set. 
        # [TODO] Need check correctness
        iCT = 0
        while iCT <= len(clickTimes):
            thisClickTime = clickTimes[:,iCT]
            tDiff = clickTimes[:,0] - thisClickTime
            echoes = (np.where(tDiff <= params.lockOut) and np.where(tDiff > 0))[0]
            delFlag[echoes] = 0 # flag close clicks in time for deletion.
            if not len(echoes): # advance to next detection
                iCT = iCT +1
            else: # or if some were flagged, advance to next true detection
                iCT = echoes[-1]+1
    
    # [TODO] Translate
    #### Remove times outside desired times, for guided detector case
    #     if params.guidedDetector:
    #         if not len(encounterTimes):
    #             # Convert all clickTimes to "real" datenums, re baby jesus
    #             sec2dnum = 60*60*24 # conversion factor to get from seconds to matlab datenum
    #             clickDnum = (clickTimes/sec2dnum) + hdr.start.dnum + datenum([2000,0,0])
    #             for itr2 = 1:size(clickDnum,1)
    #                 thisStart = clickDnum(itr2,1)
    #                 thisEnd = clickDnum(itr2,2)
    #                 afterStarts = find(encounterTimes(:,1)> thisStart)
    #                 firstAfterStart = min(afterStarts) # this is the start of the guided period it should be in
    #                 beforeEnd = find(encounterTimes(:,2)> thisEnd)
    #                 firstBeforeEnd = min(beforeEnd)
    #                 if firstAfterStart ~= firstBeforeEnd
    #                     # Then this click does not fall within an encounter, chuck it
    #                     delFlag(itr2) = 0
    #                 end
    #             end
    #         else
    #             fprintf('No times to prune.\n')
    #         end
    #     end

    clickTimesPruned = clickTimes[delFlag==1] # apply deletions

    np.savetxt(outFileName, clickTimesPruned, fmt='%.5f')
    return delFlag