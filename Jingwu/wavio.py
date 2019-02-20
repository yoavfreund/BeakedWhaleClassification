from __future__ import division

import io
import os
import re
import math
import struct
import numpy as np
from datetime import date
from datetime import datetime
from datetime import timedelta

def io_readXWavHeader(filename):
    hdr = type('AnonymousClass',(object,),{})()
    hdr.fType = 'xwav'

    with open(filename, 'rb') as fh:
        ##################################################
        # RIFF chunk
        ##################################################
        riff, size, fformat = struct.unpack('4sI4s', fh.read(12))

        hdr.xhd = type('AnonymousClass',(object,),{})()

        hdr.xhd.ChunkID   = riff    # "RIFF"
        hdr.xhd.ChunkSize = size    # File size - 8 bytes
        hdr.xhd.Format    = fformat

        filesize = os.path.getsize(filename)
        if hdr.xhd.ChunkSize != filesize - 8:
            raise Exception('Error - incorrect Chunk Size')

        if not (hdr.xhd.ChunkID == 'RIFF' and hdr.xhd.Format == 'WAVE'):
            raise Exception('not wav file - exit')

        ##################################################
        # Format Subchunk
        ##################################################
        hdr.xhd.fSubchunkID   = struct.unpack('4s', fh.read(4))[0]  # "fmt "
        hdr.xhd.fSubchunkSize = struct.unpack('I', fh.read(4))[0]   # (Size of Subchunk - 8) = 16 bytes (PCM)  
        hdr.xhd.AudioFormat   = struct.unpack('H', fh.read(2))[0]   # Compression code (PCM = 1)
        hdr.xhd.NumChannels   = struct.unpack('H', fh.read(2))[0]   # Number of Channels
        hdr.xhd.SampleRate    = struct.unpack('I', fh.read(4))[0]   # Sampling Rate (samples/second)
        hdr.xhd.ByteRate      = struct.unpack('I', fh.read(4))[0]   # Byte Rate = SampleRate * NumChannels * BitsPerSample / 8
        hdr.xhd.BlockAlign    = struct.unpack('H', fh.read(2))[0]   # # of Bytes per Sample Slice = NumChannels * BitsPerSample / 8
        hdr.xhd.BitsPerSample = struct.unpack('H', fh.read(2))[0]   # of Bits per Sample : 8bit = 8, 16bit = 16, etc

        if not (hdr.xhd.fSubchunkID == 'fmt ' and hdr.xhd.fSubchunkSize == 16):
            raise Exception('unknown wav format - exit')

        hdr.nBits = hdr.xhd.BitsPerSample       # # of Bits per Sample : 8bit = 8, 16bit = 16, etc

        hdr.samp = type('AnonymousClass',(object,),{})()
        hdr.samp.byte = math.floor(hdr.nBits/8)      # # of Bytes per Sample

        ##################################################
        # HARP Subchunk
        ##################################################
        hdr.xhd.hSubchunkID = struct.unpack('4s', fh.read(4))[0]   # "harp"
        if hdr.xhd.hSubchunkID == 'data':
            print 'normal wav file - read data now'
        elif hdr.xhd.hSubchunkID != 'harp':
            raise Exception('unsupported wav format:', hdr.xhd.hSubchunkID)
            return

        hdr.xhd.hSubchunkSize     = struct.unpack('I', fh.read(4))[0]   # (SizeofSubchunk-8) includes write subchunk
        hdr.xhd.WavVersionNumber  = struct.unpack('B', fh.read(1))[0]   # Version number of the "harp" header (0-255)
        hdr.xhd.FirmwareVersionNuumber = struct.unpack('10s', fh.read(10))[0]# HARP Firmware Vesion
        hdr.xhd.InstrumentID      = struct.unpack('4s', fh.read(4))[0]      # Instrument ID Number (0-255)
        hdr.xhd.SiteName          = struct.unpack('4s', fh.read(4))[0]      # Site Name, 4 alpha-numeric characters
        hdr.xhd.ExperimentName    = struct.unpack('8s', fh.read(8))[0]      # Experiment Name
        hdr.xhd.DiskSequenceNumber= struct.unpack('B', fh.read(1))[0]       # Disk Sequence Number (1-16)
        hdr.xhd.DiskSerialNumber  = struct.unpack('8s', fh.read(8))[0]      # Disk Serial Number
        hdr.xhd.NumOfRawFiles     = struct.unpack('H', fh.read(2))[0]       # Number of RawFiles in XWAV file
        hdr.xhd.Longitude         = struct.unpack('i', fh.read(4))[0]       # Longitude (+/- 180 degrees) * 100,000
        hdr.xhd.Latitude          = struct.unpack('i', fh.read(4))[0]       # Latitude (+/- 90 degrees) * 100,000
        hdr.xhd.Depth             = struct.unpack('h', fh.read(2))[0]       # Depth, positive == down
        hdr.xhd.Reserved          = struct.unpack('8s', fh.read(8))[0]      # Padding to extend subchunk to 64 bytes

        if hdr.xhd.hSubchunkSize != (64 - 8 + hdr.xhd.NumOfRawFiles * 32):
            raise Exception ('Error - HARP SubchunkSize and NumOfRawFiles discrepancy?')
            return

        #####################################################
        # write sub-sub chunk
        #####################################################
        hdr.raw = type('AnonymousClass',(object,),{})()
        hdr.xhd.year,         hdr.xhd.month,        hdr.xhd.day,        hdr.xhd.hour\
        ,hdr.xhd.minute,      hdr.xhd.secs,         hdr.xhd.ticks,      hdr.xhd.byte_loc\
        ,hdr.xhd.byte_length, hdr.xhd.write_length, hdr.xhd.sample_rate,hdr.xhd.gain\
        ,hdr.raw.dnumStart,   hdr.raw.dvecStart,    hdr.raw.dnumEnd,    hdr.raw.dvecEnd\
        = ([None]*hdr.xhd.NumOfRawFiles for i in range(16))
        for i in range(hdr.xhd.NumOfRawFiles):
            # Start of Raw file :
            hdr.xhd.year[i]     = struct.unpack('B', fh.read(1))[0]       # Year
            hdr.xhd.month[i]    = struct.unpack('B', fh.read(1))[0]       # Month
            hdr.xhd.day[i]      = struct.unpack('B', fh.read(1))[0]       # Day
            hdr.xhd.hour[i]     = struct.unpack('B', fh.read(1))[0]       # Hour
            hdr.xhd.minute[i]   = struct.unpack('B', fh.read(1))[0]       # Minute
            hdr.xhd.secs[i]     = struct.unpack('B', fh.read(1))[0]       # Seconds
            hdr.xhd.ticks[i]    = struct.unpack('H', fh.read(2))[0]       # Milliseconds
            hdr.xhd.byte_loc[i] = struct.unpack('I', fh.read(4))[0]       # Byte location in xwav file of RawFile start
            hdr.xhd.byte_length[i]  = struct.unpack('I', fh.read(4))[0]   # Byte length of RawFile in xwav file
            hdr.xhd.write_length[i] = struct.unpack('I', fh.read(4))[0]   # # of blocks in RawFile length (default = 60000)
            hdr.xhd.sample_rate[i]  = struct.unpack('I', fh.read(4))[0]   # sample rate of this RawFile
            hdr.xhd.gain[i] = struct.unpack('B', fh.read(1))[0]           # gain (1 = no change)
            hdr.xhd.padding = struct.unpack('7s', fh.read(7))[0]          # Padding to make it 32 bytes...misc info can be added here
            try:
                secs = int(hdr.xhd.secs[i])
            except ValueError:
                secs = 0
                
            # calculate starting time [dnum => datenum in days] for each raw
            secdelta = secs+hdr.xhd.ticks[i]/1000
            secfloat = secdelta - int(secdelta)
            dtime = datetime(hdr.xhd.year[i], hdr.xhd.month[i],  hdr.xhd.day[i],\
                             hdr.xhd.hour[i], hdr.xhd.minute[i], int(secdelta))
            frac = ((dtime-datetime(dtime.year,dtime.month,dtime.day,0,0,0)).seconds+secfloat) / (24.0 * 60.0 * 60.0)
            hdr.raw.dnumStart[i] = date.toordinal(dtime)+366+frac
            hdr.raw.dvecStart[i] = [hdr.xhd.year[i], hdr.xhd.month[i],  hdr.xhd.day[i],\
                                    hdr.xhd.hour[i], hdr.xhd.minute[i], secdelta]
            
            # end of RawFile:
            secdelta = (hdr.xhd.byte_length[i]-2)/hdr.xhd.ByteRate
            secfloat = secdelta - int(secdelta)
            dtime_end = dtime + timedelta(seconds=int(secdelta))
            frac = ((dtime_end-datetime(dtime_end.year,dtime_end.month,dtime_end.day,0,0,0)).seconds+secfloat)\
                        / (24.0 * 60.0 * 60.0)
            hdr.raw.dnumEnd[i] = date.toordinal(dtime_end)+366+frac
            hdr.raw.dvecEnd[i] = [sum(x) for x in zip(hdr.raw.dvecStart[i], [0,0,0,0,0,secdelta])]

        #########################################################
        # DATA Subchunk
        #########################################################
        hdr.xhd.dSubchunkID = struct.unpack('4s', fh.read(4))[0]     # "data"
        if hdr.xhd.dSubchunkID != 'data':
            raise Exception('hummm, should be "data" here? SubchunkID = ' + hdr.xhd.dSubchunkID)

        hdr.xhd.dSubchunkSize = struct.unpack('I', fh.read(4))[0]    # (Size of Subchunk - 8) includes write subchunk

        hdr.nch = hdr.xhd.NumChannels         # Number of Channels
        hdr.fs  = hdr.xhd.sample_rate[0]       # Real xwav Sampling Rate (samples/second)

        # vectors (NumOfWrites)
        hdr.xgain = hdr.xhd.gain[0];            # gain (1 = no change)
        hdr.start = type('AnonymousClass',(object,),{})()
        hdr.start.dnum = hdr.raw.dnumStart[0]
        hdr.start.dvec = hdr.raw.dvecStart[0]
        hdr.end = type('AnonymousClass',(object,),{})()
        hdr.end.dnum = hdr.raw.dnumEnd[hdr.xhd.NumOfRawFiles-1]
    return hdr

def io_readWavHeader(filename, dateRegExp):
        
    hdr = type('AnonymousClass',(object,),{})()
    hdr.fType = 'wav'

    with open(filename, 'rb') as fh:
        # Riff = io_readRIFFCkHdr(f_handle);
        riff, size, fformat = struct.unpack('4sI4s', fh.read(12))
        if riff != 'RIFF':
            raise Exception('io:%s is not a RIFF wave file' % filename)
            return 
        if fformat != 'WAVE':
            raise Exception('io:%s Riff type not WAVE' % filename)
            return
        
        Chunks = list()
        
        # [TODO] WHILE LOOP FOR READING CHUNK
        
        # read fmt chunk        
        Chunk = io_readRIFFCkHdr(fh)
        
        if Chunk.ID == b'fmt ':
            Chunk.Info = io_readRIFFCk_fmt(fh)
            hdr.fmtChunk = len(Chunks)  # Note fmt idx
            
        Chunks.append(Chunk)

        # read data chunk
        Chunk = io_readRIFFCkHdr(fh)

        hdr.dataChunk = len(Chunks)  # Note data idx
        Chunks.append(Chunk)

        hdr.Chunks = Chunks

        # Calculate number of samples - round number to avoid small errors
        hdr.Chunks[hdr.dataChunk].nSamples = int(hdr.Chunks[hdr.dataChunk].DataSize /\
                                       (hdr.Chunks[hdr.fmtChunk].Info.nBytesPerSample *\
                                        hdr.Chunks[hdr.fmtChunk].Info.nChannels))
        hdr.fs    = hdr.Chunks[hdr.fmtChunk].Info.nSamplesPerSec;
        hdr.nch   = hdr.Chunks[hdr.fmtChunk].Info.nChannels;
        hdr.nBits = hdr.Chunks[hdr.fmtChunk].Info.nBytesPerSample * 8;
        
        hdr.samp = type('AnonymousClass',(object,),{})()
        hdr.samp.byte = hdr.Chunks[hdr.fmtChunk].Info.nBytesPerSample;
        
        hdr.xhd = type('AnonymousClass',(object,),{})()
        hdr.xhd.ByteRate    = hdr.Chunks[hdr.fmtChunk].Info.nBlockAlign * hdr.fs;
        hdr.xhd.byte_length = hdr.Chunks[hdr.dataChunk].DataSize;
        hdr.xhd.byte_loc    = hdr.Chunks[hdr.dataChunk].DataStart;

        # [QUESTION?] NO harpChunk mentioned in HDR before
        
        # no HARP format
        # Add HARP data structures for uniform access
        hdr.xgain = 1          # gain (1 = no change)
        pureName = os.path.splitext(os.path.splitext(filename)[0])[0]
        catDate = re.search(dateRegExp, pureName).group(0).replace('_','')
        
        hdr.start = type('AnonymousClass',(object,),{})()
        if len(catDate) == 12:
            hdr.start.dvec = [int(catDate[:2])+2000, int(catDate[2:4]), int(catDate[4:6]),\
                          int(catDate[6:8]), int(catDate[8:10]), int(catDate[10:12])]
        elif len(catDate) == 14:
            hdr.start.dvec = [int(catDate[:4]), int(catDate[4:6]), int(catDate[6:8]),\
                          int(catDate[8:10]), int(catDate[10:12]), int(catDate[12:14])]
        else:
            raise Exception('Error:%s has wrong date format' % filename)
            return
        
        hdr.xhd.year    = hdr.start.dvec[0];          # Year
        hdr.xhd.month   = hdr.start.dvec[1];         # Month
        hdr.xhd.day     = hdr.start.dvec[2];           # Day
        hdr.xhd.hour    = hdr.start.dvec[3];          # Hour
        hdr.xhd.minute  = hdr.start.dvec[4];        # Minute
        hdr.xhd.secs    = hdr.start.dvec[5];          # Seconds
        
        dtime = datetime(hdr.xhd.year, hdr.xhd.month, hdr.xhd.day,\
                         hdr.xhd.hour, hdr.xhd.minute, hdr.xhd.secs)
        hdr.start.dnum = date.toordinal(dtime)+366

        samplesN = hdr.xhd.byte_length / (hdr.nch * hdr.samp.byte)
        hdr.end = type('AnonymousClass',(object,),{})()
        hdr.end.dnum = date.toordinal(dtime + timedelta(seconds=samplesN/hdr.fs))+366
    return hdr

def io_readRIFFCkHdr(fh):
    # [TODO] handle EOF
    Chunk = type('AnonymousClass',(object,),{})()
    Chunk.StartByte = fh.tell()
    Chunk.ID = struct.unpack('4s', fh.read(4))[0]
    HeaderSize = 8  # Oy vay - Magic
    Chunk.DataSize = struct.unpack('I', fh.read(4))[0]
    # Beginning of chunk data
    Chunk.DataStart = fh.tell()
    Chunk.ChunkSize = Chunk.DataSize + HeaderSize
    return Chunk

def io_readRIFFCk_fmt(fh):
    # Given a handle to a file positioned at the first data byte
    # of a RIFF format chunk, read the format information.
    Fmt = type('AnonymousClass',(object,),{})()
    # Data encoding format
    Fmt.wFormatTag      = struct.unpack('H', fh.read(2))[0]
    # Number of channels
    Fmt.nChannels       = struct.unpack('H', fh.read(2))[0]
    # Samples per second
    Fmt.nSamplesPerSec  = struct.unpack('I', fh.read(4))[0]
    # Avg transfer rate
    Fmt.nAvgBytesPerSec = struct.unpack('I', fh.read(4))[0]
    # Block alignment
    Fmt.nBlockAlign     = struct.unpack('H', fh.read(2))[0]
    
    Fmt.fmt = type('AnonymousClass',(object,),{})()
    if Fmt.wFormatTag == 1:
        # PCM Format
        Fmt.fmt.nBitsPerSample = struct.unpack('H', fh.read(2))[0]
        # [TODO] check bytes remaining io_RIFFCk_BytesRemainingP
    else:
        raise Exception('io:Unsupported codec:  %s', Fmt.wFormatTag)
        return
    
    # Determine # of bytes per sample
    Fmt.nBytesPerSample = int(math.ceil(Fmt.nBlockAlign/Fmt.nChannels));
    
    # Type of data can be determined by the number
    # of bytes per sample.
    if Fmt.nBytesPerSample == 1:
        Fmt.fmt.dtype = 'uchar' # unsigned 8-bit
    elif Fmt.nBytesPerSample == 2:
        Fmt.fmt.dtype = 'int16' # signed 16-bit
    elif Fmt.nBytesPerSample == 3:
        Fmt.fmt.dtype = 'bit24' # signed 24-bit
    elif Fmt.nBytesPerSample == 4:
        # Check format tag to see whether signed 32-bit or
        if Fmt.wFormatTag == 1:
            Fmt.fmt.dtype = 'int32' # signed 32-bit
        elif Fmt.wFormatTag == 3:
            Fmt.fmt.dtype = 'float' # normalized floating point
        elif Fmt.wFormatTag == 4:
            Fmt.fmt.dtype = 'float' # floating point
        else:
            raise Exception('io:Unsupported wFormatTag', Fmt.wFormatTag)
    
    # Handle special case for 24 bit data
    if Fmt.wFormatTag != 3 and Fmt.fmt.nBitsPerSample == 24:
        Fmt.BytesPerSample = 3
    return Fmt

def io_readRaw(fh, hdr, rawNum, channels):
    # function data = io_readRaw(fh, hdr, rawNum, channels)
    # Given a handle to an open XWAV file and header information,
    # retrieve the data in raw file number rawNum.
    #
    # When multiple channels are present, only returns the first channel
    # unless the user specifies which channels should be returned
    # (e.g. channels 2 and 4: [2 4]).
    
    # [FUTURE] How to parallel file read?

    if hdr.fType == 'xwav':

        if hdr.nBits == 16:
            dtype = np.int16
        elif hdr.nBits == 32:
            dtype = np.int32
        else:
            raise Exception('hdr.nBits = ', hdr.nBits, 'not supported')
            return
        
        
        # Will hdr.xhd.byte_length always be the same across files
        # samples, byte_loc, byte_length
        samples = int(hdr.xhd.byte_length[rawNum] / (hdr.nch * hdr.samp.byte))
        
        fh.seek(hdr.xhd.byte_loc[rawNum])
        data = np.fromfile(fh, dtype, hdr.nch*samples).reshape((hdr.nch, samples))
        # [TODO] Check data empty
        if hdr.xhd.gain[rawNum] > 0:
            data = data / hdr.xhd.gain[rawNum]
        data = data[channels-1,:]  # Select specific channel(s)
    return data