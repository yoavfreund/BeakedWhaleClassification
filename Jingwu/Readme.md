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

#### Two-phase process:

Phase 1: The goal is to identify and extract consistent features of clicks

- Identify positively detected echolocation clicks into five-minute bins in testing set.

- Clustering was performed on the spectra of those clicks. Similarity metric $S_{SPEC}$ was used to mark the distance between each two spectra nodes. Weak linked edges were dropped out (~95%). Clusters were produced using CW clustering algorithm.

  *Some exploratory analysis is conducted here to prune the dropout rate. Most of bins contain a single cluster, 5.57% bins contain more than one clusters*

- Generate summary node(denote C_i), mean spectral levels and modal ICIs from its cluster of clicks, for each cluster that produced by previous step.

Phase 2: the goal is to identify recurrent click types across many bins

+ A combined similarity metric (S2) consisting of both spectral and ICI information from summary node was computed

+ Clustering was performed among those summary nodes. S2 was used to mark the distance between each two summary nodes. Weak linked edges were dropped out (~95%). Clusters were produced using CW clustering algorithm across 20 iterations. 

  *The exploratory analysis suggested that a pe value of 0.95 led to stable partitions with minimal isolation and few overly-trained or duplicate clusters.*

+ Seven dominant and recurrent click types characterized by consistent spectral shapes and modal ICIs were identified, aka, seven dominant clusters were produced in phase 2. 

Classification:

​	FOR each test summary node Ci

​		FOR each click type Tj

​			compare Ci to all of the training nodes of click type Tj

​		ENDFOR

​		obtain a similarity metric

​		assign Ci to Tj

​	ENDFOR	

## Questions:

1. What is "CV" mentioned in the paper?

   In Phase 1, the average number of automatically identified clusters per time bin ranged from 1.02 to 1.14 (CV = 0.14 and 0.35 respectively) across sites and deployments (Table 2). 

2. Why he used "5-minute bin"?

3. How does the method classify isolated clicks(nodes) during clustering from phase 1 or 2?

4. Why classification only uses summary nodes from phase 1? 

5. True or False? The method clustered seven dominant and recurrent click types after phase 2 using only test data? 

6. True of False? The method classified each summary node into one of the click type clusters T from the training data?

7. Is above two steps duplicated? Since it clustered seven click types(unlabeled), and then it tried to label each summary node to specific click type using training data.

