*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Stroke Prediction 

The aim of the exercise is to create two sets of deliverables a) AutoML implementation  and b) Hyperdrive implementation of stroke prediction. 
Though term is "prediction", this is a classification problem. This answers the basic question - given the characteristic of the person
is this person likely to have stroke. The features and labels are stated below. 

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

Data is registered as shown below

![images](https://github.com/Vzard/Assignment-2/blob/main/images/image005.png)


### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
