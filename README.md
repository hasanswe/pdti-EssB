# Prediction of Drug-Target Interaction Based on Protein Features Using  Undersampling and Feature Selection Techniques with Boosting

# Abstract
Accurate identification of drug-target interaction (DTI) is a crucial and challenging task in the drug
discovery process, having enormous benefit to the patients and pharmaceutical company. The
traditional wet-lab experiments of DTI is expensive, time-consuming, and labor-intensive.
Therefore, many computational techniques have been established for this purpose; although a huge
number of interactions are still undiscovered. Here, we present pdti-EssB, a new computational
model for identification of DTI using protein sequence and drug molecular structure. More
specifically, each drug molecule is transformed as the molecular substructure fingerprint. For a
protein sequence, different descriptors are utilized to represent its evolutionary, sequence, and
structural information. Besides, our proposed method uses data balancing techniques to handle the
imbalance problem and applies a novel feature eliminator to extract the best optimal features for
accurate prediction. In this paper, four classes of DTI benchmark datasets are used to construct a
predictive model with XGBoost. Here, the auROC is utilized as an evaluation metric to compare
the performance of pdti-EssB method with recent methods, applying five-fold cross-validation.
Finally, the experimental results indicate that our proposed method is able to outperform other
approaches in predicting DTI, and introduces new drug-target interaction samples based on
prediction probability scores.

# Benchmark Datasets:

(I) Enzyme (II) Ion Channel (III) GPCR (IV) Nuclear Receptor



# Feature Generation and Experiments codes

Protein Feature Extraction: 

Evolutionary Based Features: PSSM bigram => X

Bigram Main.m and Evolutionary Based Features Bigram PSSM.m files was used to create the feature group X

Sequence Features: PseAAC => Y

PseAAC.py was used to create the feature group Y 

Structural Based Features: SSC, ASAC, TAC,TAAC, SPAC, TAB, SPB => Z

Structural Based Features-SPIDER2 (1), Structural Based Features-SPIDER2 (2) and Structural Based Features-SPIDER2 (3) files was used to create the feature group Z

Drug Feature Extraction:

Structural based Properties: MSF  => A

CalculatePubChemFingerprint.py file was used to create the feature group A


Random Under Sampling.py and Cluster Under Sampling.py files was used for to balance the datasets. 
 
Feature Selection.py file was used to reduce the drug-target features.

Effect of Feature Groups and Classifiers.py and Effect of Balancing Methods.py files was used for test the effect of different Feature Groups, Classifiers and Balancing
techniques on different datasets. 

[Same code can be applied for different datasets with different features]









