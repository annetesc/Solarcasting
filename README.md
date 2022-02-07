# Solarcasting
Code for the scientific paper 'Spatio-Temporal Resolution of Irradiance Samples in Machine Learning Approaches for Irradiance Forecasting', available at https://ieeexplore.ieee.org/document/9035506

Code for Spatio-temporal resolution of irradiance samples in Machine Learning approaches for irradiance forcasting.

### Workflow
1.data_manager: downloads the raw data from inforiego ftp-server for a specified time period for all stations of the network;
2.csv_manager:	concatenates the individual csv files from selected stations to a single csv file;
3.data_preparation_Manager: cleans data, adds coordinates, adds GHI and solar angles from clear sky solar model Mc Clear, generates relative radiation, removes outliers with relative radiation >1.5; splits data set into X and y files and creates a y_persist file with the data for the baseline model (scaled persistence);
4.train_test_set: generates training and test data sets; specifies disjunct periods for training and testing, selects the features (between 1 and 4); specifies the prediction horizon(s) (0.5 hour, 1, 1.5, 2, 3 or 4 hours); the data is shuffled and stored in a specified folder for training and test data;
5.gridsearch: gridsearch uses a 10-fold crossvalidation method on the training data; the user can add or remove hyperparameters via the config_file, enter  possible values for every parameter as a list and thus defines the grid for the gridsearch. The algorithm finds the best combination of those hyperparameters. 
6.train: trains a selected model (“ANN”: Multilayer Perceptron, “ARX”: Linear Regression, “RT”: Regression Tree, “RRF”: Random Forest Regression); for every forecast horizon two models are trained: one with all features from every station included and a local model with just the features from the target station; 
for “ARX” and “RRF” a separate file with the individual importances of all the features is created and stored for later use (plotting figures);
7. predict: evaluates the pretrained models (for all horizons) on the test data set; a text file with the results is generated and stored in a specified folder; the metrics include: RMSE for non-local and local model, RMSE for baseline model (scaled persistence) and SKILLS for both non-local and local model;
8. map_info_riego:	plots a map with all the available ubications from the inforiego network;
9. plot_importances_stations_ARX:	evaluates the coeffinciencts of the linear regression to show the importance of the individual stations for the training of the “ARX”-model (using sum of all features from a station);
   plot_importances_stations_RRF:	evaluates the feature importances of the Random Forest regression to show the importance of the individual stations for the training of the “RRF”- model (using sum of all features from a station).
   plot_importances_features_ARX:	evaluates the coeffinciencts of the linear regression to show the single most important feature from every station and its importance for the overall model (“ARX”);
   plot_importances_features_RRF:	evaluates the feature importances of the Random Forest regression to show the single most important feature from every station and its importance for the overall model (“RRF”);   
   
  ### Requirements
- Python 3.7.4
- Pandas 
- Numpy
- Scikit-Learn 0.20




