*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Stroke Prediction 

The aim of the exercise is to create two sets of deliverables a) AutoML implementation  and b) Hyperdrive implementation of stroke prediction. 
Though term is "prediction", this is a classification problem. This answers the basic question - given the characteristic of the person
is this person likely to have had a stroke. The features and labels are stated below. 

*TODO:* Write a short introduction to your project.

## Project Set Up and Installation

There are no special setup required. I have data downloaded into by github. This is accessed via code in Azure ML. 
https://raw.githubusercontent.com/Vzard/ML-Azure-Capstone/master/StrokeData_V1.csv

## Dataset

### Overview
The data set is from Kaggle at http:///www.kaggle.com. The data has following features and label, and values

1. Age - positive integer
2. gender	 - Male/Female
3. hypertension	 - 1/0 (yes/No)
4. heart_disease	 -1/0 (Yes/No)
5. ever_married	- (yes/No)
6. work_type	- (children (less than 18 years)/never worked/self employed/Govt/Private)
7. Residence_type	- Urban/rural
8. avg_glucose_level	( real number)
9. bmi	(real number)
10. Smoking_status	(unknown/formerly smoked/smokes/Never Snoked)
=========================================================
12. stroke (1/0) -label - This the the decision variable. 
 
### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access

Data is accessed via TabularDatasetFactory.from_delimited_files(path=url_path)

## Automated ML

### Data is registered as shown below

![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/registerdata.png)

### Parameters and explanations

automl_settings = {
    "experiment_timeout_minutes": 20,  // This sets timeout for experiment
    "max_concurrent_iterations": 4,  //4 trials run concurrently
    "primary_metric" : 'AUC_weighted', // This is chosen given the imbalanced data
    "n_cross_validation" :10  //lowest empirically acceptable values. 
}
 
automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",  //type of ML model
                             training_data=dataset,
                             label_column_name="stroke",  //label
                             enable_early_stopping= True,
                             test_size =0.2,  // Total is about 5000 data points, 1000 chosen for training
                             featurization= 'auto',  //did not go EDA, since point of exercise to showcase Azure ML aspect
                             debug_log = "automl_errors.log",
                             **automl_settings

###  Snapshot of model in progress towards deployment

### - Models set to run
![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/Models_running.png)

### Run details
![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/rundetails.png)

![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/rundetails_2.png)

### Best Model

![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/best_model.png)

###Registered Model

![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/new_registered_model.png)

### Deployed Model
![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/newmodel_deployed.png)

### Post deployment state
![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/post_dep_state.png)

### Using the endpoint
![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/using_the_service.png)

### JSON ooutput
![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/jsonputput.png)

### Results

Accuracy
0.95132
(primary metric)
AUC weighted
0.85832

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.
### Posted the run detail widget and Posted the best model

## Hyperparameter Tuning

Logistic regression that lends easy for classification purposes is used. And it also provides parsimonious parameters

parameter_sampling = RandomParameterSampling(
                    {
                        "--C":uniform(0.05,0.10),
                        "--max_iter":choice(25,50,75,100) 
                    }
)

#for ML method 2
# Create a SKLearn estimator for use with train.py

#TODO: Create your estimator and hyperdrive config

sklearn_env = Environment.from_conda_specification('sklearn-env',"conda_dependencies.yml")

hyperdrive_run_config = HyperDriveConfig(run_config=estimator,
                                         hyperparameter_sampling=parameter_sampling,
                                         primary_metric_name="Accuracy",
                                         primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                         max_total_runs=25,
                                         max_concurrent_runs=5)

## Screenshots

## Model Deployment

AutoML model deployed (best model) stated in AutoML section

### Deployed Model
![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/newmodel_deployed.png)

### Post deployment state
![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/post_dep_state.png)

### Using the endpoint
![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/using_the_service.png)

### JSON ooutput
![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/jsonputput.png)

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
