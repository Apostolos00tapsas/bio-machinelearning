# bio-machinelearning
Aim:
This project was created with aim to train three diferent models(Linear Regretion Model, Artificial Neural Network and a 
Support Vector Machine) which will predict if someome has skin cancer(Greek: Melanoma). 

Data and Methods:
UK biobank data contain 488.377 genotypes. 
From 438.427 has been genotyped to 804.427 polymorphisms from AffymetrixUK  Biobank  Axiom  Array  chip and the other 49.950 has been 
genotyped to 807.411 polymorphism from Affymetrix  Affymetrix  UK  BiLEVE  Axiom  Array  chip for the UK  BiLEVE study. 
The imputation register was held from UK Biobank with UK10K 1000 Genomes phase 3 platform and Consortium Reference Consortium platform.
Of the 488,377 British Biobank participants with available genetic data, people of non-European origin were excluded and using 
centrally-provided data from the English family of Biobank, exclusions from the analysis were any pairs of 1st and 2nd tier grades.
Instances of melanoma have been identified by linking to the central registries of the National Health Center of the United Kingdom (NHS).
NHS core registers provide information on cancer registries and deaths coded in accordance with the 10th revision of the International 
Classification of Diseases (ICD-9 (172) and ICD-10 (C43) (WHO, http: // www .who.int / classifications / icd / el /).
We ended up with 2,871 cases of cancer from people of European origin whose first diagnosis of cancer was skin cancer (melanoma). 
Of the remaining UK Biobank participants, 378,624 healthy controls were selected who had never been diagnosed with cancer or had reported 
self-reported cancer and had never been registered with the national cancer registry.
Then, 564 patients who were diagnosed with melanoma and 564 healthy patients were selected to study an amount of models  
that would predict whether a new patient was going to develop skin cancer (melanoma).

Codes:
(The coding genomics file)
In this type of problems we have two type of data, numerical data and genomic data witch must be converted in 0,1 or 2.
We are conver the snps_file baset on the effective allele collumn of snps_and_effect_allele file selected duos as:
For instance, we select the first double T and G whose effective allele is G, thus mean we must code it to 1. 
So we have 4 diferent cases. 1- Both are equal with the effective allele, thus mean we have must code it to 2.
2- One of the double is equal with the effective allele, thus mean we have 1. 3- None of the double is equal so we code it to 0. 
4- We have a missing value, thus mean that the double is 0 and 0, and we code it to miss.
After that procedure we are merge both numerical(sex,age..) and genomic(A1,A2..) data into the Dataset file. 
(The Train model file)
In this file we are training the models. We are not using any pre-process method like PCA because of known predictive value of characteristics.


