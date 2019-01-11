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

2. How does the method classify isolated clicks(nodes) during clustering from phase 1 or 2?

3. True of False? Classification in testing data also requires phase 1 which produces summary nodes, and then match those summary nodes to click types found in training data.

4. True of False? The method classifies a cluster of nodes(summary node) to one the seven click types, instead of classifying every node to one click type in testing data. 

