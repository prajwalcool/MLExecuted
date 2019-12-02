# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 20:34:19 2019

@author: Prajwal
"""
import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
'thal', 'heartdisease']
heartDisease = pd.read_csv('data7_heart.csv', names = names)
heartDisease = heartDisease.replace('?', np.nan) 
model = BayesianModel([('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('exang',
'trestbps'),('trestbps','heartdisease'),('fbs','heartdisease'),('heartdisease','restecg'),
('heartdisease','thalach'), ('heartdisease','chol')])
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)
HeartDisease_infer = VariableElimination(model)
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 65})
print(q)