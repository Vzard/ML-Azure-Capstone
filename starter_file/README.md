# video link


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

The data is imbalanched and hence the choice of AUC as a primary metric.  AutoML does well with 0.85
## Hyperparameter Tuning

Logistic regression that lends easy for classification purposes is used. And it also provides parsimonious parameters

parameter_sampling = RandomParameterSampling(
                    {
                        "--C":uniform(0.05,0.10),
                        "--max_iter":choice(25,50,75,100) 
                    }
)

###Create your estimator and hyperdrive config

sklearn_env = Environment.from_conda_specification('sklearn-env',"conda_dependencies.yml")
estimator = ScriptRunConfig(source_directory='.',
                            command=['python', 'train.py'],
                            compute_target=compute_target,
                            environment=sklearn_env)

hyperdrive_run_config = HyperDriveConfig(run_config=estimator,
                                         hyperparameter_sampling=parameter_sampling,
                                         primary_metric_name="Accuracy",
                                         primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                         max_total_runs=25,
                                         max_concurrent_runs=5)


## Screenshots
### Hyper drive job

![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/hyper_drive_job.png)

### hyper drive best model detail
![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/best_model_detail.png)

### Overview of best model
![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/ hyper_fina_best.png)

### Registered

![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/hyper_drive_register.png)

## Model Deployment - (AutoML deployed)
AutoML model deployed (best model) stated in AutoML section

### Deployed Model
![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/newmodel_deployed.png)

### Post deployment state
![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/post_dep_state.png)

### Using the endpoint
![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/using_the_service.png)

### JSON output
![images](https://github.com/Vzard/ML-Azure-Capstone/blob/master/starter_file/jsonputput.png)

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
Clearly, use of SMOTE to resolve imbalanced data could have enabled AUC to a better value. Random Forest could have been an alternate choice. To a great extent, attempted to increase cross validation in AutoML to reduce the impact of balanced data. The hyperdrive run did not include such methods. 
