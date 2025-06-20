# EPA-simmodel
# README File

Created by: EPA141 Group 

|    Name     | Student Number |
| :---------: | :------------- |
| Amaryllis Brosens| 5307554       |
|  Celia Martínez Sillero  | 6222102         |
| Daniela Ríos Mora | 6275486       |
| Oscar Rajantie Tuñon| 5477913        |



## Introduction

In the folder (MBDM-Group 1),  can find place group 1's lab submission for the final project of EPA141 Model-based decision-making course.

This README file is to help a first-time user understand what it is about and how they might be able to use it.
 

## Purpose and objective of this project

The purpose of this project is to assist the Transport Company (TC) during the negociations and decision-making process of the Room for the River project. This project is located in the IJssel river, and concerns the provinces of Gelderland and Overijssel.

This project uses model-based techniques to support the Transport Company. It is noteworthy that this work is done by the analysts of the TC and not the TC themselves. Thereby, the results are meant to be independent and reliable, as the analyst team do not have a political agenda.

## Structure

The following submission is provided with the following structure:
Structure inside this ZIP file: 

    Folder MBDM-Group 1

        Subfolder Results
            Subfolder fw_shapes
            -Final_policies_experiments.csv
            -Final_policies_outcomes.csv
            -Results Filter.xlsx
            -Robustness_experiments.csv
            -Robustness_outcomes.csv

        Subfolder data
            -Subfolder fragcurves
            -Subfolder hydrology
            -Subfolder losses_tables
            -Subfolder muskingum
            -Subfolder rating_curves
            -EWS.xlsx
            -dikeIjssel.xlsx
            -dikeIjssel_alldata.xlsx
            -reference_scenario
            -reference_scenario.csv
            -rfr_strategies.xlsx   

        Subfolder output
            -Subfolder fw_shapes
            -convergence_seed_27097.csv
            -convergence_seed_45646.csv
            -convergence_seed_567.csv
            -convergence_seed_676465.csv
            -convergence_seed_90.csv
            -results_seed_27097.csv
            -results_seed_45646.csv
            -results_seed_567.csv
            -results_seed_676465.csv
            -results_seed_90.csv
            -seed_27097_archive.tar.gz
            -seed_45646_archive.tar.gz
            -seed_567_archive.tar.gz
            -seed_676465_archive.tar.gz
            -seed_90_archive.tar.gz

        -Convergence Metrics.ipynb
        -DB_Optimization.py
        -DB_Optimization_fw_shapes.py
        -MORDM Analysis fw_shape_0.ipynb
        -MORDM Analysis fw_shape_110.ipynb
        -MORDM Analysis fw_shape_132.ipynb
        -MORDM Analysis fw_shape_22.ipynb
        -MORDM Analysis fw_shape_44.ipynb
        -MORDM Analysis fw_shape_66.ipynb
        -MORDM Analysis fw_shape_88.ipynb
        -MORDM Analysis.ipynb
        -MORDM PRIM.ipynb
        -MORDM.xlsx
        -Open_exploration_scenario_policy.ipynb
        -Problem formulation.ipynb
        -Results_filter
        -dike_model_function.py
        -dike_model_optimization.py
        -dike_model_simulation.py
        -funs_dikes.py
        -funs_economy.py
        -funs_generate_network.py
        -funs_hydrostat.py
        -problem_formulation.py
        -rfr_IJssel.png
        -rf_wlreduction.png

 
    -README.md

      Subfolder report
            -MBDM_Group 1_Report.pdf
            -MBDM_Group 1_Political Reflection.pdf



      -README.md (this file)


## Main Files and Folders

 File/Folder                     | Purpose                                                                                                                                  |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `Results/`                        | Stores the experiments, outcomes, filtered policies and final robust policies produced by MORDM Analysis.ipynb C                                            |
| `data/`                  | Contains the data with the reference scenario produced by PRIM.ipynb, the functions used by the model to perform calculations and the RfR strategies.                                                        |
| `output/`                      | Stores the outputs of the optimization algorithm and the convergence metrics in Optimization_DB.py runned for multiple seeds.                                                                                 |
| `Results/`                     | Stores the experiments and outcomes produced by PRIM, including robustness metrics and final policies.  
| `Open_exploration_scenario_policy.ipynb`  | Contains the code for the BAU analysis with random trees and sobol. It generates the reference scenario found with PRIM.               
| `DB_Optimization.ipynb` | Runs the optimization algorithm across multiple seeds and generates corresponding convergence results.                             |
| `Convergence Metrics.ipynb` | Calculates and plots convergence metrics derived from the optimization results.                                                      |
| `MORDM Analysis.`  | Generates the Pareto front from the optimization results, calculates the robustness metrics, and identifies the top 10 candidate strategies.                                     |
| `MORDM PRIM.ipynb`    |       Applies scenario discovery to the candidate solutions                 |



## How to Use

To reproduce the analysis and explore the results, follow the steps below. This guide assumes a working knowledge of Python and Jupyter Notebooks.

### 1. Set up your environment

- Ensure you are using **Python 3.8 or higher**.
- Install required packages:ema_workbench 2.5.3 Mesa 2.1.4
