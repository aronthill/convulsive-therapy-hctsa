# Supplementary analysis code

Code accompanying the manuscript:

**Uncovering Neural Signatures of Convulsive Therapy in Depression Using Massive EEG Time-Series Feature Extraction**  

Much of the study used existing EEG and time-series analysis toolboxes. EEG artefact cleaning was performed using [RELAX](https://github.com/NeilwBailey/RELAX) within EEGLAB. Feature extraction, descriptive feature ranking, clustering and associated visualisations used functions provided by the [hctsa toolbox](https://github.com/benfulcher/hctsa). Please refer to these repositories for the toolbox code, installation instructions and relevant citations.

This repository provides additional study-specific code supporting the manuscript's paired statistical comparisons, pre/post classification and exploratory treatment-response prediction. These scripts complement the toolbox-based analyses; they do not contain every analysis or figure-generation step in the manuscript. They continue to call hctsa functions where indicated, with additional MATLAB and Python code for the paired tests and participant-level validation procedures.

## Files and run order

| Script | Purpose |
|---|---|
| A_paired_feature_tests.m | Paired pre/post feature comparisons with FDR correction. |
| B_prepare_training_pca.m | Fit PCA using training participants only. |
| C_extract_fold_features.m | Calculate hctsa features from the PC signals for each validation fold. |
| D_prepost_classification.m | Five-fold pre/post classification and within-participant label permutations. |
| E_select_response_features.m | Select representative features using training participants only. |
| F_response_prediction.py | Leave-one-participant-out treatment-response prediction and permutation testing. |

A runs independently, for each PC. Run B then C separately for five-fold and leave-one-participant-out validation. D uses the five-fold outputs; E then F use the leave-one-participant-out outputs. The `support` folder contains helper code required by these scripts. `feature_inventory.csv` lists the retained and selected features.

## Using the code

Edit the file paths near the top of each script for your computer. The scripts require the study's prepared EEG, hctsa files and participant information, which are not included. **EEG and clinical data cannot be shared because of ethics restrictions.**

The revised analyses used MATLAB R2025b with hctsa and the required MATLAB toolboxes; Python dependencies are listed in `requirements.txt`. Settings and random seeds are specified in the scripts. These simplified scripts start fresh and do not support pause/resume. Calculations have been checked against saved study results, but a complete fresh installation and rerun of this package have not been verified. Further methodological details are provided in the manuscript and supplementary materials.
