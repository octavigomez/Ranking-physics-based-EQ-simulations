# Ranking-physics-based-EQ-simulations

This repository contains the datasets supplementary to the publication "A multi-benchmark performance analysis of physics-based earthquake simulations" submitted to Geophysical Research Letters.

The datasets include the codes to run the ranking analyses, inputs and outputs for the RSQSim earthquake simulation cases explained in the paper: a single fault and the fault system of the Eastern Betics Shear Zone (simulations from Herrero-Barbero et al. 2021). The results and data are stored in a separate folder for each case study presented in the paper: "Single fault" and "EBSZ". Each folder contains a series of subfolders and a Python script to run the ranking analysis for that specific case study. The script contains the default path references to read all necessary input files for the analysis and automatically save all the outputs. The subfolders are:

- ./Inputs: This folder contains the input files required for the RSQSim simulations. This includes:
  
  a. The fault model ("Nodes_RSQSim.flt" and "EBSZ_model.csv" for the single fault and EBSZ cases, respectively), which specifies the coordinate nodes of the fault triangular meshes and fault properties such as rake (º) and slip rate (m/yr).
  b. Neighbor file ("neighbors.dat"/"neighbors.12") that contains lists of triangular patches of the fault model that are neighboring. This file is used in RSQSim.
  c. Input parameter file ("Input_Parameters.txt"): this file specifies the parameters that are variable in each catalogue. This file is just for information purposes and is not used for the calculations.
  d. Parameter file(s) to run the RSQSim calculations.
    *For the single fault, this file is common ("test_normal.in") and is updated during the calculation when executing the "Run.sh" file in the terminal to run RSQSim. This file contains a script that loops through the input parameters a, b and normal stress explored in the study and changes the input parameter file accordingly in each iteration.
    *For the EBSZ, this file is specific for each simulation ("param_EBSZ_(n).in"), as each simulation was run separately (not as a batch process).
  e. (Only for the EBSZ case) Input paleoseismic data for the paleorate benchmark. One file ("coord_sites_EBSZ.csv") contains a list of UTM coordinates of each paleoseismic site in the EBSZ and another ("paleo_rates_EBSZ.csv") contains the mean paleoearthquake rates and 1 sigma uncertainties in those sites (data from Herrero-Barbero et al., 2021).
  
- ./Simulation_models: contains several subfolders, one for each simulated catalogue (96 for the single fault case and 12 for the EBSZ). Each subfolder contains data that is read by the Ranking Python code to perform the analysis.
  *For the single fault, the folder names follow the structure "model_(normal stress)_(a)_(b)".
  *For the EBSZ, the folder names are "cat-(n)".
- ./Ranking_results: contains the outputs of the ranking analysis, which are two figures and one text file.
  *Figure 1 ("Final_ranking.pdf"): visualization of the final ranking analysis for all models against the analyzed benchmarks.
  *Figure 2 ("Parameter_sensitivity.pdf"): visualization of the final and benchmark performance versus the input parameter of the models.
  *Text file ("Ranking_results.txt"): contains the final and benchmark scores of each simulation model. This file is outputted so the user can reproduce and customize their own figures with the ranking results.

To use the ranking codes in you own datasets, please replicate the folder structure explained above. Use the code that best suits your data: use the one for the single fault if you wish not to use the paleorate benchmarks, and use the EBSZ one if you wish to include these data in your analysis. At the beginning of the respective codes (before the "Start" block comment) you will find the variables where the file names of the fault model and paleoseismic data are indicated. Change them to adapt it to your data. There you can also assign weights to the respective benchmarks in the analysis (default is set at 1).
