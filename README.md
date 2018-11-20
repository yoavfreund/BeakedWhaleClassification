# WhaleClassification
Project whose goal is the automatic classification of Whales and Dolphins from recordings of their underwater sounds.  Toothed whales production sterotyped [echolocation clicks](https://en.wikipedia.org/wiki/Animal_echolocation#Toothed_whales) and baleen whales produce sounds that are sometimes called [songs](https://en.wikipedia.org/wiki/Whale_vocalization).  Examples of whale and dolphin sounds can be found at the [Voices in the Sea](http://cetus.ucsd.edu/voicesinthesea_org/). 

## notebooks 
To be added by Yoav.

## Data
Data is stored on two buckets in S3

1. `s3://gulf-whales`: Contains underwaters sound clips of echolocation clicks from two kinds of beaked whales (Cuvier's and Gervais') that were recorded in the Gulf of Mexico after the Deepwater Horizon oil spill. The goal is a binary classification that separates these two species.
   * [Two page description](https://docs.google.com/document/d/1GYivLB5e4xM-URTivAGFOqcjyXp-Ay8s_fyRSTcHvL0/edit#heading=h.lnna1gml3l15)

2. `s3://hdsi-whales`: 4TB of sound data from the Pacific Ocean and 4TB of data from the Atlantic Ocean which were annotated for whale and dolphin sounds for the Marine Mammal Detection, Classification, Localization and Density Estimation Workshops (DCLDE) that were conducted in: 
   * 2015 [7th International DCLDE](http://www.cetus.ucsd.edu/dclde/) which is based on marine mammal sounds in the **Pacific**  
   * 2018 [8th International DCLDE](http://sabiod.univ-tln.fr/DCLDE/) which is based on marine mammal sounds in the **Atlantic** 
   * Listing of files is in `hdsi-whales.ls`

## Kait Frasier's method
[Kate Frasier](https://www.researchgate.net/profile/Kaitlin_Frasier) recently published a new approach for [Automated classification of dolphin echolocation click types](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005823) based on unsupervised network-based classification that has achieved excellent results for classifiying a variety of species.

## Proposed Project Steps
### Preparatory Steps
1. Write a high level summary of click descriptions including the following terms: click detector, click spectra, inter-click-interval, click bout, peak-to-peak amplitude, and click time series. 
2. Perform low level signal processing including click detection and definition of an amplitude threshold, spectral calculations and principal-component-analysis (PCA). 

### Replication of Prior Results
3. Re-do the notebooks for the Gulf of Mexico beaked whale dataset.



