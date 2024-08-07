# Dskzinopnc

This project contain `source code` of ACT and `revised manuscirpt`.

The results from this demo might slightly differ from those reported in our manuscript (usually better).

We'll continuously refine the code of ACT, and align results in the paper with actual code.

****
For this demo, only tablet dataset is prepared

If you want try other datasets, please download datasets and write corresponding data loaders (see Page 13 revised manuscirpt).

****
__Dataset details__

>__Tablet__ dataset is originally proposed in IDRC shoot-out. Tablet contains NIR measurements of pharmaceutical tablets from two spectrometers (referred as spectrometer No. 1 and No. 2 in this paper), ranging from 600 $\mathrm{nm}$ to 1898 $\mathrm{nm}$. The spectra are collected from 655 pharmaceutical tablets from production runs and pilot runs. Each spectrum contains 650 sampling points with an interval of 2 $\mathrm{nm}$. The content of active pharmaceutical ingredient (API) varies from 80\% to 120\% of target value (195 mg), following the requirements of International Conference on Harmonization (ICH) and the requirements of U.S. Food and Drug Administration (FDA). This dataset is divided into calibration set (155 tablets), validation set (40 tablets), and test set (460 tablets). The aim of this dataset is to predict the amount of API (in mg) within the tablets.

>__Melamine__ (MF) dataset originates from a batch-condensation process at Metadynea GmbH (Krems, Austria) and consists of NIR spectra from different MF recipes with slightly different compositions. The spectra covers the first and second overtone regions, which are located at wavenumbers at 5546 $\mathrm{cm}^{-1}$- 6254 $\mathrm{cm}^{-1}$ (1803 $\mathrm{nm}$ - 1598 $\mathrm{nm}$) and 6596 $\mathrm{cm}^{-1}$ – 6975 $\mathrm{cm}^{-1}$ ( 1433 $\mathrm{nm}$ - 1516 $\mathrm{nm}$). The spectra are recorded in 346 spectral bands. Two recipes, namely R562 and R568, are included in this paper. There are 3032 samples in R562, while 733 samples in R568. When using a recipe for training, 70\% samples are used as training set and the rest 30\% are used as validation set. The analytical target of this dataset is turbidity point which indicates the degree of polymerization.

>__Mango\_DMC__ dataset consists of 11,691 spectra collected from 4675 mango samples across 112 populations and 4 seasons. The first three seasons are used for training (7413) and validating (2830), while the last season is used for testing (1448). Spectra are collected with a F750 Produce Quality Meter, ranging from 300 $\mathrm{nm}$ to 1100 $\mathrm{nm}$. There is a 3 $\mathrm{nm}$ interval between neighboring spectral bands (242 sampling points), while the optical resolution is 10 $\mathrm{nm}$. The analytical task of this dataset is to predict the dry matter content (DMC) of mango fruit. DMC is an index of total carbohydrates (starch and sugars) and correlates strongly to the Soluble Solids Content (SSC) of ripened fruit, which influences the eating flavor of mango fruits. DMC can also be used a harvest maturity guide, in conjunction with flesh color.

>__Strawberry__ dataset contains 983 MIR fruit purees collected by Fourier transform infrared (FTIR) spectrometer with attenuated total reflectance (ATR) sampling. Spectra are recorded with 235 data points ranging from 899 $\mathrm{cm}^{-1}$ to 1802 $\mathrm{cm}^{-1}$. Among the spectra, 337 spectra are used for training, 329 spectra are used for validation, and 317 spectra are used for testing. The single-beam spectra of the purees were ratioed to background spectra of water and then converted into absorbance units. Infrared spectroscopy is expected to replace the slow and expensive chemical analyses. The analytical task of this dataset is to detect adulteration in strawberry purees, where the samples are divided into two classes: strawberry purees and adulterated strawberry purees.

>__Apple\_leaf__ dataset contains 5,490 NIR spectra collected from the leaves of apple trees covering 20 different varieties. Training set consists of 2,500 spectra, validation set consists of 1,250 spectra, while testing set contains 1,740 spectra. Each apple leaf is measured by ten times, deriving ten spectra respectively. The spectrometer utilized in this dataset is ASD FieldSpec 3, which records spectra ranging from 300 $\mathrm{nm}$ to 2500 $\mathrm{nm}$. Spectral resolution between 300 $\mathrm{nm}$ and 1000 $\mathrm{nm}$ is 3 $\mathrm{nm}$, while spectral resolution between 1001 $\mathrm{nm}$ and 2500 $\mathrm{nm}$ is 6 $\mathrm{nm}$. 300 spectral bands ranging from 1000 $\mathrm{nm}$ to 2500 $\mathrm{nm}$ are included in the experiments. The analytical task of this dataset is to classify the apple leaves from different varieties.
