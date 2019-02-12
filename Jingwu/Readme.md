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

**Workflow:**

1. read params(settings)
2. read header
3. build filter
4. bandpass data 
   - this is actually high pass, where threshold is **5000** Hz
5. calculate threshold
   - Is this fixed by setting params, or is there a calculation?
6. LR (First pass): extract click candidates by thresholding the filtered signal
   - Is there max-min(p2p) calculation involved?
7. HR (Second pass) expand click region
   - It seems this is done using iterative expansion, is that really necessary or can we just take a window of fixed size window around the detected peak. Yoav suggests this will be faster and the steps making the window are not symmetric or merging peaks can be done afterwards, because all of the relevant information is kept and only things that are far from the peak are eliminated.
8. prune out candidate, remove windows where peaks are too high
9. compute params
   - large number of params are computed for each window 
   - Is it possible to restrict this step to computing params that are only currently used by Kait to collect clicks (or say for classification use).
   - 

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