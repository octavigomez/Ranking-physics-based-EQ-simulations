# Method for ranking physics based earthquake simulations

This repository contains the codes to perform the ranking analysis of physics-based earthquake simulation models explained in the article "A multi-benchmark performance analysis of physics-based earthquake simulations", currently submitted for publication.

In this repository we provide basic structure file and codes to run the analysis that will be mantained and updated over time. We also provide a sample of datasets for users to test the codes and perform an example analysis. These convey a sample of the cases explained in the paper: a single fault and the fault system of the Eastern Betics Shear Zone (simulations from Herrero-Barbero et al. 2021). For full datasets and frozen-version codes of the publication please visit the related Zenodo repository: 

Folder structure: 

We provide two folders, one containing the code and structure to run the analysis with paleoeartquake data (Analysis_w_paleodata) and another without them (Analysis_wo_paleodata). Each folder contains a series of subfolders and a Python script to run the ranking analysis for that specific case study. The script contains the default path references to read all necessary input files for the analysis and automatically save all the outputs. The subfolders are:

./Inputs: This folder contains the input files required for the analysis:

a. The fault model ("Nodes_RSQSim.flt" and "EBSZ_model.csv" for the single fault and EBSZ cases shown in the paper, respectively), which specifies the coordinate nodes of the fault triangular meshes and fault properties such as rake (º) and slip rate (m/yr).

b. (Only for the case with paleodata) Input paleoseismic data for the paleorate benchmark. One file ("coord_paleosites.csv") contains a list of UTM coordinates of each paleoseismic site and another ("paleo_rates.csv") contains the mean paleoearthquake rates and 1 sigma uncertainties in those site. The example data is for the EBSZ (from Herrero-Barbero et al., 2021).

./Simulation_models: contains several subfolders, one for each simulated catalogue. Each subfolder contains data that is read by the Ranking Python code to perform the analysis.

./Ranking_results: When running the code this folder is created and the outputs of the ranking analysis are stored there, which are two figures and one text file: 

*Figure 1 ("Final_ranking.pdf"): visualization of the final ranking analysis for all models against the analyzed benchmarks.

*Figure 2 ("Parameter_sensitivity.pdf"): visualization of the final and benchmark performance versus the input parameter of the models.

*Text file ("Ranking_results.txt"): contains the final and benchmark scores of each simulation model. This file is outputted so the user can reproduce and customize their own figures with the ranking results.

To use the ranking codes in you own datasets, please replicate the folder structure explained above. Use the code that best suits your data: use the one inside "Analysis_wo_paleodata" if you wish not to use the paleorate benchmarks, and use the "Analysis_wo_paleodata" if you wish to include these data in your analysis. Do not forget to replace the input files with the data you wish to analyze. At the beginning of the respective codes (before the "Start" block comment) you will find the variables where the file names of the fault model and paleoseismic data are indicated. Change them to adapt it to your data. There you can also assign weights to the respective benchmarks in the analysis (default is set at equal weights).
