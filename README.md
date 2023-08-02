# Interpretable features 

* All supporting functions are in "functions" folder. Please see the details for each function file in README.md within the folder

execute files are listed here:

* **run_extract_trial.py** given the start and stop information, extract the trial from edf or asc files. Parameters:
  * path input and output
  * reextract, reprocess
  * which trials
  * analysis_constants, have tested, but change if needed
 
* **run_processing.py** get the eye movement and extract general eye movement features. save plots and summary csv as needed. Parameters:
  * path input and output
  * reextract, reprocess, save_processed, save_csv, save_plots, show_plots
  * which trials
  * analysis_constants, have tested, but change if needed

* **run_saccade.py** . get the saccade analysis and extract saccade tasks features. save summary csv as needed. Parameters:
  * path input and output
  * reextract, reprocess, save_processed, save_csv
  * which trials
  * analysis_constants, have tested, but change if needed
  * anti: anti-saccade or not

* **run_summary.py** given summary csv, plot the boxplot and  calculate statistics. Parameters:
  * path input and output
  * use_trialContaining, plot_summaryBoxplots, plot_rawBoxplots, plot_correlations, pvalue_threshold, boxplot_orde

* **run_misc.py** miscellaneous activities such as:
  * synchronize data files from OneDrive to local folder 
  * horizontal versus. vertical saccades ratio
  * calculate the mean and std of trails for general eyetracking feature in saccade tasks
  * extract trial_info csv
  * select the features that are statistically significant

* **run_classification.py** use xgboost and cross validation on extracted features for classification. Parameters:
  * source path 
  * task_list 

* **run_classify.py** use xgboost, random forest, k-NN on extracted features for classification or k-means, tsne for clustering. Parameters:
  * source path 
  * normalize_data,  transform_data, transofrm_whiten_data, transformation
  * print_all_accuracy, predict_kmeans, predict_xgboost, predict_knn, predict_randomForrest
 
* **Longitude_analysis.py** given the feature summary csv, perform longitude analysis for patients. Parameters:
  * input path, output path 
 
* **xgb_multiclass_custom_softmax.py** a test demo for multi-class xgboost. Parameters:
  * kRows, kCols, kClasses, kRounds
