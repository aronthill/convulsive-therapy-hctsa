# Supplementary analysis code

Code accompanying the manuscript:

**Uncovering Neural Signatures of Convulsive Therapy in Depression Using Massive EEG Time-Series Feature Extraction**  

Much of the study used existing EEG and time-series analysis toolboxes. EEG artefact cleaning was performed using [RELAX](https://github.com/NeilwBailey/RELAX) within EEGLAB/MATLAB. Feature extraction, descriptive feature ranking, clustering and associated visualisations used functions provided by the [hctsa toolbox](https://github.com/benfulcher/hctsa). Please refer to these repositories for the toolbox code, installation instructions and relevant citations.

This repository provides additional study-specific code supporting the manuscript's paired statistical comparisons, pre/post classification and exploratory treatment-response prediction. These scripts complement the toolbox-based analyses; they do not contain every analysis or figure-generation step in the manuscript. They continue to call hctsa functions where indicated, with additional MATLAB and Python code for the paired tests and participant-level validation procedures.

 **Please note, EEG and clinical data cannot be shared because of ethics restrictions.**

The revised analyses used MATLAB with hctsa and the required MATLAB toolboxes; Python dependencies are listed in `requirements.txt`. Further methodological details are provided in the manuscript and supplementary materials.
