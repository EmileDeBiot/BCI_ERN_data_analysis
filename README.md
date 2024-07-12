# BCI_ERN data analysis

## Pre-processing

To pre-process your data, use the pre_processing_viz.ipynb notebook. 
Here are the steps of preprocessing:
1. Pass-band filtering (0.3 - 40 Hz)
2. Noisy channel Interpolation
3. Independent Component Analysis + Removal of none brain components with iclabel (machine learning)


## Error-related-potential analysis

To analyse error-related potentials related to the BCI flankers task, use the erp_analysis.ipynb notebook.
Here are the steps of the analysis:
1. Import events and focus on the feedback trigger event
2. Separate in 2 groups based on the feedback type
3. Compute mean evoked response for each group