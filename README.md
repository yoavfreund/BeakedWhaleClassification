# WhaleClassification
Project whose goal is the automatic classification of Whales and Dolphins from recordings of their underwater sounds.  Toothed whales production sterotyped [echolocation clicks](https://en.wikipedia.org/wiki/Animal_echolocation#Toothed_whales) and baleen whales produce sounds that are sometimes called [songs](https://en.wikipedia.org/wiki/Whale_vocalization).  Examples of whale and dolphin sounds can be found at the [Voices in the Sea](http://cetus.ucsd.edu/voicesinthesea_org/). 

## Notebooks

1. SETUP:

   **[InitialSetup.ipynb](https://github.com/yoavfreund/BeakedWhaleClassification/blob/master/Sumit_et_al/InitialSetup.ipynb)**: Install Git; Clone existing repo; Copy Data.

2. DATA ANALYSIS:

   **[Data_Processing_Whales.ipynb](https://github.com/yoavfreund/BeakedWhaleClassification/blob/master/DSE230_version/Data_Processing_Whales.ipynb)**: includes a detailed description of echo-location clicks data fields, and some sample plots of waveforms and spectra for both species. **Two species generate different waveforms.** **peak2peak** measures the difference between the max and the min of the wave form. **Spectra** is computed from the waveform using FFT. Applying PCA on spectra, the top 5 eigenvectors explain ~85% variances. Still, Overlaps exist on the projection of top eigenvectors of two species. It is not clear how eigenvectors of spectra can be used to distinguish two species.

   >  Q: What could be the interpretation of eigenvectors?

   **[XGBoost_Whales.ipynb](https://github.com/yoavfreund/BeakedWhaleClassification/blob/master/DSE230_version/XGBoost_Whales.ipynb)**: compares the feature's relative importance among first 10 eigen vectors, rmse and peak2peak of spectra, by fitting into an XGBModel. The **first 2-3 eigenvectors** turn out to be the most important, usually assigned at least ~20% more weights.

   > Q: How well is XGBModel fitted onto this classification task? What is its accuracy?

   **[Training and Feature Extraction with Reassigned Labels - ICI Mode, Peak2Peak, RMSE, Eigen.ipynb](https://github.com/GrEedWish/BeakedWhaleClassification/blob/label_based_on_majority_vote/Training%20and%20Feature%20Extraction%20with%20Reassigned%20Labels%20-%20ICI%20Mode%2C%20Peak2Peak%2C%20RMSE%2C%20Eigen.ipynb)**: introduces **Interclick Interval(ICI)**, the time difference between two clicks within a given bout. **Two species have obvious differences on the overall distribution of ICI**. **Mode** of ICI has more distinguishable distribution than the median of ICI between two species.

3. MODEL

   **[Training and Feature Extraction - ICI Mode, Peak2Peak, RMSE, Eigen.ipynb](https://github.com/yoavfreund/BeakedWhaleClassification/blob/master/Sumit_et_al/Training%20and%20Feature%20Extraction%20-%20ICI%20Mode%2C%20Peak2Peak%2C%20RMSE%2C%20Eigen.ipynb)**: designs a feature vector which includes **1)PCA projection values of spectra by taking the first 5 eigenvectors,  2) rmse of spectra, 3)peak2peak, and 4)ICI Mode.** The goal is to optimize the prediction accuracy of species with objective function:

   $obj(\theta) = \frac{\sum_y I(y, \hat y)}{\sum_y}$

   Five classification models are applied to truly detected and correctly classified data samples.

   |                   | Logistic Regression | SVM Model | Decision Tree | Random Forest | GB Trees |
   | ----------------- | :-------------------------: | :---------: | :-------------: | :-------------: | :--------: |
   | Training Accuracy | 0.8302                    | 0.8303    | 0.8544        | 0.8447        | **0.8574** |
   | Testing Accuracy  | 0.8301                    | 0.8301    | 0.8542        | 0.8445        | **0.8572** |

   **[Training and Feature Extraction-ICI median.ipynb](https://github.com/yoavfreund/BeakedWhaleClassification/blob/master/Sumit_et_al/Training%20and%20Feature%20Extraction-ICI%20median.ipynb)**: uses all same features except for ICI Median instead of ICI Mode. The classification gives similar performance for five different models.

   > Q: Why ICI Median, which is less representative than Mode, generates similar model?

   **[Training and Feature Extraction with Reassigned Labels - ICI Mode, Peak2Peak, RMSE, Eigen.ipynb](https://github.com/GrEedWish/BeakedWhaleClassification/blob/label_based_on_majority_vote/Training%20and%20Feature%20Extraction%20with%20Reassigned%20Labels%20-%20ICI%20Mode%2C%20Peak2Peak%2C%20RMSE%2C%20Eigen.ipynb)**:

   > Q: What does relabeling do?

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
4. Re-write Kate Frasier's code in Python.
5. Replicate the click type analysis found in Kate's paper (note that only a subset of all the Atlantic species were found in the Gulf of Mexico and visa-versa).

### Possible Direction for Novel Project Analysis
7. Use the annonated data sets for the DCLDE 2015 and 2018 to discover the full range of sounds present.
8. Explore alternative unsupervised learning methods.
9. Explore supervised learning methods such as boosting, and random forest.



