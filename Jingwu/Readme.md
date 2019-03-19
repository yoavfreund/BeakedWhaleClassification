## Clicks:

Toothed whales use clicks for echolocation, such as sensing surroundings and detecting predators. The clicks can be separated into “runs/bouts” of clicks, with each click bout usually corresponding to a single species. Different species of beaked whales emit clicks with different waveforms and spectral distributions, where click spectrum is computed from the waveform within a given specific click time series using DFT. Some exploratory analysis also suggests that inter-click-interval provides distinguishable information across different species. Another information, peak-to-peak amplitude, is also extracted from the click waveform (not sure why).

The clicks data comes from five click detectors located in Gulf of Mexico at three continental slope and two shelf locations between 2010 and 2012. The goal is classifying Whales and Dolphins from recordings of their underwater echo-location clicks. 

## Terms:

- **Click detector:** devices with algorithm filters that identify impulse signals to detect echolocation clicks.

- **Click spectra:** click pressure level as a function of frequency, is computed from the click waveform using FFT. 

- **Inter-click-interval:** the time difference between two clicks.

- **Click bout:** the clicks can be separated into "bouts" of clicks, where each bout usually corresponds to a single species.

- **Peak-to-peak amplitude:** the difference between the max and the min of the wave form.

- **Click time series:** an interval of click waveform, which is transformed to click spectra using discrete Fourier transform (DFT). Certain samples before and after each detected click were included in the click time series.
- **Click-positive bin:** bins containing at least 100 click detections.
- **modal ICI**: the first peak in the ICI distribution, aka, the most occurrence ICI amongst a cluster of clicks.
- **Summary nodes**: Representative nodes(clicks) extracted from testing set, which need to be classified into specific click type by comparing to training data based on similarity.

## Summary of Kait Frasier's method:

#### Two-phase automated clustering process:

Phase 1: The goal is to identify and extract consistent features of clicks

- Clustering was performed on the spectra of echolocation clicks into five-minute click-positive bins.

- Similarity metric $S_{SPEC}$ was used to calculate the distance between each two spectra clicks. Weakly linked edges were dropped out (~95%).  $S_{spec}$ is a function of pairwise normalized spectral row vectors.

  *Some exploratory analysis is conducted here to prune the dropout rate -- consistent to manual reviews. Most of bins contain a single cluster, 5.57% bins contain more than one clusters*

- Generate summary nodes(denote C_i), mean spectral levels and modal ICIs from its cluster of clicks, for clusters identified in each bin.

Phase 2: the goal is to identify recurrent click types across many bins, produce template clusters for classification

+ Clustering was performed on the summary spectra and ICI distributions.

+ A combined similarity metric (S2) consisting of both spectral and ICI information from summary node was computed. S2 is a product of pairwise modal ICI similarity $S_{ICI}$ and Spectral similarity $S_{spec}$.

  *The exploratory analysis suggested that a pe value of 0.95 led to stable partitions with minimal isolation and few overly-trained or duplicate clusters.*

+ Seven dominant and recurrent click types were identified in the training datasets across five sites by the automated clustering method. 

The automated clustering process from training data produced seven click type clusters(denote T_j) that will be used in classification for testing data.

#### Classification:

The goal is to find the best click type match, among seven click types that were discovered from training data, for summary node in test data.

~~~
FOR each test summary node Ci
	FOR each click type Tj
		compute the spectrum and modal ICI for Ci
		compare Ci to all nodes in click type Tj based on the similarity  
	ENDFOR
		obtain a similarity metric
		assign test node Ci to click type Tj
ENDFOR
~~~

## Questions:

1. What is "CV" mentioned in the paper?

   In Phase 1, the average number of automatically identified clusters per time bin ranged from 1.02 to 1.14 (CV = 0.14 and 0.35 respectively) across sites and deployments (Table 2). 

2. True of False? In training data, we first generate summary nodes following phase1, and then cluster those summary nodes into 7 major click types. In testing data, we first generate summary nodes following phase1, and then match those summary node into one of the click type that we found using training data.

3. Instead of doing clustering in phase1, where we isolated small portion (7.4%) of nodes, to get a summary node, could we just use the modal of all nodes (excluding outliers) to be the summary node? What could be the trade off between runtime and accuracy?

   - For the cases where a bin has more than 2 clusters, we try to treat it as one cluster because first its not common (5.7%), second it should not affect too much.
   - For the cases where a bin has more than 5000 nodes, we can split it into k bags, where which bag selects its own summary node

> The reason for doing this is phase 1 is basically producing one summary node per bin, where almost all nodes are preserved. While phase 2 is actually where CW algorithm do have the effects of clustering nodes (into 7 categories).

4. Why phase 1 uses (only) the spectra of echolocation clicks as weighting/scoring matrix?

   Would cooperating with other features (like p2p) helps generate better weighting matrix that leads to better summary node here? 

5. Can we extract more information from summary node that helps the clustering in phase2? Such as summary cluster variance, summary cluster confidence or widest difference within summary cluster (like spectra energy of a species usually comes with a range, the range of summary cluster may be helpful here)?

6. In phase 2, does other information, like click time, detection location, sequential relationship between bins(summary node), could be joined into the scoring matrix to get better performance? `S2 = S_ICI 􏰛 * S_SPEC ` --> `S2 = f(S1, S2, S3, ... S_i)`

7. `Types that were predominantly restricted to shallow sites`. Could the type variance is largely caused by the shallow sites noise? (Apart from obvious false positive samples we removed beforehand)



1. How to synchronize reading and processing wav files in Python?
2. I couldn't regenerate the same LR results from XWAV file that Kait provided.



## Detector

**Terms:**

Count: the raw measurement from the sensor

Low Resolution: First pass which detect many candidates which above a certain threshold

High Resolution: Second pass which filters out candidates

> High resolution step also more accurately identifies detection starts and ends.

**Workflow:**

1. read params(settings)
2. read header
3. build filter
4. bandpass data 

   - this is actually high pass, where threshold is **5000** Hz

   > The filtering needs vary by use case. I generally only use a high pass, but people often use a band pass to more narrowly focus their detections on target signals and reduce false positives. 
5. calculate threshold

   - Is this fixed by setting params, or is there a calculation?

   > Either. It was originally specified in params, and still can be in my code, but I think it is better to calculate it automatically based on the desired received level in dB. This is a little hard if we are using a very wide detection band (like a 5 kHz high pass) because the sensor frequency response is not totally flat. Depending on what transfer function value you use to convert to counts, you can get very different answers. Ideally you would use the minimum transfer function value, but if you do that, you get way too many detections to handle, so I have generally selected the mean or median transfer function value across the band of interest.
6. LR (First pass): extract click candidates by thresholding the filtered signal

   - Is there max-min(p2p) calculation involved?

   > No, this is just based on energy (signal^2) peaks exceeding the counts threshold
7. HR (Second pass) expand click region

   - It seems this is done using iterative expansion, is that really necessary or can we just take a window of fixed size window around the detected peak. Yoav suggests this will be faster and the steps making the window are not symmetric or merging peaks can be done afterwards, because all of the relevant information is kept and only things that are far from the peak are eliminated.

   > I think we could just take a fixed window, but we should think about what happens if there are more than one signals in the window. The optimal window size would change based on the signal (e.g. longer for beaked whales, shorter for dolphins). So if we use a long window, then we are potentially batching signals and under-counting detections. A short window might cut signals off. So it's not immediately clear to me what the best solution there would be. At the very least, the window length should be modifiable by the user.
8. prune out candidate, remove windows where peaks are too high
9. compute params
   - large number of params are computed for each window 
   - Is it possible to restrict this step to computing params that are only currently used by Kait to collect clicks (or say for classification use).

   > Yes. We talked about this a little before, but the things we use later are detection times, waveforms, peak2peak amplitudes, and spectra

## Milestone:

+ 1/21 - 1/25
  + Study FFT
  + Further read the paper
  + Experiments with basic signal processing methods
  + Plan to start to project

+ 1/26 - 2/3
  + Met with Kait, John and Yoav, discussed project overview
  + Read through Kait's code about detector, replayed her code in Matlab
  + Further read on [AudioFiltering.ipynb](https://github.com/calebmadrigal/FourierTalkOSCON/blob/master/09_AudioFiltering.ipynb)
  + Started translating code into python, (processing header)

+ 2/3 - 2/10
  + Get runnable file samples from Kait. (tf is important here)
  + Translated code (finished Header, LR, HR)
  + [Q] Ask prof. how to parallel things

+ 2/10-2/18
  + Finished majority translation from .m to .py
  + Got runnable program and similar results
  + Investigated on the time cost for each function 

+ 2/18-2/23
  + 2nd check on the click detector outputs
  + Finished the code organization, with runnable py file

+ 2/23-3/3
  + [Choices on how to select filters](https://dsp.stackexchange.com/questions/9467/what-is-the-advantage-of-matlabs-filtfilt)
  + [Filter Types](https://community.plm.automation.siemens.com/t5/Testing-Knowledge-Base/Introduction-to-Filters-FIR-versus-IIR/ta-p/520959)
  + Butterworth is one type of IIR, which is typically faster than FIR to achieve the similar results

+ 3/3 -3/14

  + Python

    <img src='outputs/runtime_1.png'>

  + Matlab

    <img src='outputs/matlab_runtime_1.png'>



| AMI        | CPU  | MEMORY | COST             | 1              | 2              | 3              | 4               | 5    | 6    | 7    |
| ---------- | ---- | ------ | ---------------- | -------------- | -------------- | -------------- | --------------- | ---- | ---- | ---- |
| t2.medium  | 2    | 4 GiB  | $0.0464 per Hour | Memory Error   |                |                |                 |      |      |      |
| t3.large   | 2    | 8 GiB  | $0.0832 per Hour | 21s<br/>40MB/s | 32s<br/>50MB/s | 40s<br/>60MB/s | 58s <br/>55MB/s |      |      |      |
| a1.xlarge  | 4    | 8 GiB  | $0.102 per Hour  | 26s            | 26s            | 30s            | 37s             |      |      |      |
| c5.xlarge  | 4    | 8 GiB  | $0.17 per Hour   | 17.5s          | 18s            | 23s            | 27s             | 32s  | 37s  | 42s  |
| r5d.large  | 2    | 16     |                  | 18s            | 24s            | 36s            | 48s             |      |      |      |
| c5.2xlarge | 8    | 16     |                  | 17s            | 18s            | 18s            | 18s             |      | 22s  | 24s  |

| Model   | t2.medium | t3.large  | a1.xlarge | c5.xlarge  | r5d.large | c5.2xlarge |
| ------- | --------- | --------- | --------- | ---------- | --------- | ---------- |
| CPU     | 2         | 2         | 4         | 4          | 2         | 8          |
| MEMORY  | 4         | 8         | 8         | 8          | 16        | 16         |
| COST    | $0.0464/h | $0.0832/h | $0.102/h  | $0.17/h    | $0.144/h  | $0.384/h   |
| 1 File  | ME        | 40MB/s    | 32MB/s    | 45MB/s     | 44MB/s    | 45MB/s     |
| 2 Files |           | 50MB/s    | 64MB/s    | 88MB/s     | 67MB/s    | 88MB/s     |
| 3 Files |           | 60MB/s    | 80MB/s    | 104MB/s    | 67MB/s    | 133MB/s    |
| 4 Files |           | 55MB/s    | 86MB/s    | 117MB/s    | 67MB/s    | 178MB/s    |
| 5 Files |           |           | ME        | 125MB/s    |           | 222MB/s    |
| 6 Files |           |           |           | ME ~130    |           | 218MB/s    |
| 7 Files |           |           |           | 130MB/s    |           | 233MB/s    |
| 8 Files |           |           |           |            |           | *          |
| Best    |           | 0.22 TB/h | 0.30 TB/h | 0.45 TB/h  | 0.24 TB/h | *0.84 TB/h |
| COST    |           | $0.39 /TB | $0.34 /TB | $0.38 / TB | $0.6 /TB  | *$0.46 /TB |

*: need more data, at least, cost can be cheaper using spot instances

ME: memory error, all the results are rounded. Results are computed from an average of two experiments.

c5.xlarge: computation optimized node

![resources](outputs/resources.png)