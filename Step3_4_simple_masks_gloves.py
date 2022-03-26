#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import functools
import matplotlib.pyplot as plt

from ema_workbench import (save_results, Constraint, SequentialEvaluator, TimeSeriesOutcome,
                           RealParameter, ScalarOutcome, CategoricalParameter, ema_logging, 
                           perform_experiments, Policy, IntegerParameter)

from ema_workbench.connectors.vensim import VensimModel
from ema_workbench.connectors import vensim
from ema_workbench.em_framework.parameters import Scenario
# from ema_workbench.analysis import parcoords


# In[2]:


if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO )


# In[3]:


from os import getcwd
wd = getcwd()


# In[4]:


#import model
Model = VensimModel('Model', wd= wd, model_file= 'Model/Infection Model Testing 02032022 Test.vpmx')


# In[5]:


#uncertainties demand 
Model.uncertainties = [RealParameter('Gloves changes per patient per day ICU',85,250),
    RealParameter('Gloves changes per patient per day Non ICU',40 ,120),
#     RealParameter('Gowns changes per patient per day ICU',10, 30),
#     RealParameter('Gowns changes per patient per day non ICU',10, 30),
    RealParameter('Simple mask changes per patient per day ICU',5, 15),
    RealParameter('Simple mask changes per patient per day non ICU',5, 15),
#     RealParameter('N95 respirators changes per patient per day ICU',2, 6),
#     RealParameter('N95 respirators changes per patient per day non ICU',1.3 , 3.9),
#     RealParameter('Eye protection changes per patient per day ICU',3, 9),
#     RealParameter('Eye protection changes per patient per day non ICU',1.3 , 3.9),

    # uncertainties forecast type
#     RealParameter('Time horizon for forecast',7 , 30),

    # uncertainties infection model
    # RealParameter('Infectivity',0.02, 0.2),

    # domestic production PPE
#     RealParameter('Transportation time domestic production PPE',1, 14),
#     RealParameter('Base raw material procurement eye protection domestic production',20000,240000),
    RealParameter('Base raw material procurement simple masks domestic production',20000,900000),
#     RealParameter('Base raw material procurement N95 respirators domestic production',5000,50000),
#     RealParameter('Base raw material procurement gowns domestic production',20000,240000),
    RealParameter('Base raw material procurement gloves domestic production',144000,1440000),
    RealParameter('Shipment time domestic production PPE',1,21),
#     RealParameter('Production time domestic production PPE',1,10),
    RealParameter('Time to reach maximum production capacity PPE dom production',5, 90),
    RealParameter('Time to reach maximum procurement capacity PPE dom production',5, 90), 
                        

    # direct tender PPE
#     RealParameter('Transportation time direct tender PPE',1,21),
#     RealParameter('Threshold to start direct tender PPE',1,21),
    RealParameter('Time to reach max direct tender',1,90),
    RealParameter('Maximum prod direct tender PPE',1,12),
#     RealParameter('Base raw material eye protection capacity direct tender',20000, 400000),
    RealParameter('Base raw material simple masks capacity direct tender',50000, 2000000),
#     RealParameter('Base raw material N95 respirators capacity direct tender',10000, 100000),
#     RealParameter('Base raw material gowns capacity direct tender',20000, 400000),
    RealParameter('Base raw material gloves capacity direct tender',800000, 50000000),
    RealParameter('Shipment time direct tender PPE',14, 120),
    RealParameter('Share of faulty PPE',0, 0.5),

                        
    #innovation PPE

    RealParameter('Reach PPE',0, 300),
    RealParameter('Production time Innovation PPE',1, 7),
#     RealParameter('Transportation time PPE innovation',3.5, 21),
    RealParameter('Shipment time innovation PPE',3.5, 14),
#     RealParameter('Base raw material eye protection capacity innovation PPE',500, 30000),
    RealParameter('Base raw material simple masks capacity innovation PPE',3400, 34000 ),
#     RealParameter('Base raw material N95 respirators capacity innovation PPE',3400, 25200),
#     RealParameter('Base raw material gowns capacity innovation PPE',500, 30000 ),
    RealParameter('Base raw material gloves capacity innovation PPE',5000, 160000 ),
#     RealParameter('Average time to approve and develop PPE',15, 120),
    RealParameter('Time to reach maximum production capacity PPE innovation',5, 90),
    RealParameter('Time to reach maximum procurement capacity PPE innovation',5, 90),                     


    # procurement worldwide PPE
#     RealParameter('Base raw material procurement eye protection worldwide',1000000,8000000),
    RealParameter('Base raw material procurement simple masks worldwide',10000000,60000000),
#     RealParameter('Base raw material procurement N95 respirators worldwide',120000,1200000),
#     RealParameter('Base raw material procurement gowns worldwide',1000000,8000000),
    RealParameter('Base raw material procurement gloves worldwide',200000000,1400000000),
    RealParameter('Preparation shipment PPE production worldwide',1, 10),
#     RealParameter('Threshold for export restriction PPE',1000000, 100000000),
    RealParameter('Delayed shipment time',30, 360),
#     RealParameter('Normal shipment time',7, 45),
    RealParameter('Reduction export PPE',0, 1),
    
    RealParameter('Maximum increase in procurement capacity PPE',1, 20),
    RealParameter('Time to reach maximum procurement capacity PPE worldwide',14, 210),
#     RealParameter('Maximum days in backlog before increase in procu capacity',1, 45),       
    RealParameter('Maximum increase in production capacity PPE',1, 20),
    RealParameter('Time to reach maximum production capacity PPE worldwide',14, 210),
#     RealParameter('Maximum days in backlog before increase in prod capacity',1, 20),   
#     RealParameter('Share of PPE ready for previous order',0.2, 1),
    RealParameter('Maximum transportation time PPE procurement world market',1, 20),
    RealParameter('change in transportation time PPE',7, 60)]


# In[6]:


Model.levers = [IntegerParameter('Switch procurement world market PPE',0,1),
               CategoricalParameter('Switch direct tender PPE',(0,1)),
               IntegerParameter('Switch domestic production PPE',0,1),
               IntegerParameter('Switch innovation PPE',0,1),
#                 IntegerParameter('Switch stockpile ventilators',0,1),
#                 IntegerParameter('Switch procurement world market ventilator',0,1),
#                 IntegerParameter('Switch direct tender ventilators',0,1),
#                 IntegerParameter('Switch innovation process ventilator',0,1),
#                 IntegerParameter('Switch loaning ventilators',0,1),
#                 IntegerParameter('Switch domestic production ventilators',0,1),
               
               # #DecisionFramework
               RealParameter('Delay domestic production PPE',7, 60),
               RealParameter('Direct tender set up time PPE',7, 45),
               RealParameter('Set up time procurement PPE worldwide',14, 50),
               RealParameter('Setting up innovation process PPE',10, 45), 
               RealParameter('Time to check PPE',1, 5),
               RealParameter('Shipment time to hospitals PPE',1, 10),
               RealParameter('Number of patients',10, 500),
               RealParameter('Preparation time for Delivery PPE',1,10),
               RealParameter('Delivery time of PPE stockpiling',1,15),
               RealParameter('Days in Stock',7, 30),
               RealParameter('Order buffer procurement world market PPE',0.5, 3),
               RealParameter('Order buffer direct tender PPE',0.5, 3),
               RealParameter('Order buffer domestic production PPE',0.5, 3),
               RealParameter('Order buffer innovation PPE',0.5, 3),
               RealParameter('Time to check products',1, 5),
               RealParameter('Shipment time to hospitals',1, 10),

               # RealParameter('Order buffer procurement world market vent',0.5, 3),
               # RealParameter('Order buffer direct tender vent',0.5, 3),
               # RealParameter('Order buffer domestic production',0.5, 3),
               # RealParameter('Order buffer innovation',0.5, 3),
               # RealParameter('Time to establish loaning process',3.5 , 21),
               RealParameter('Time horizon for forecast',5 , 30),
               # #Managing Stockpile
               RealParameter('Share of products expiring per day',0, 0.0016667),
               RealParameter('Share of stockpile available to hospitals',0, 1),
               RealParameter('Inital value for eye protection in stockpile UK',0, 78300000),#3 times advised value
               RealParameter('Inital value for simple masks in stockpile UK',0, 468000000),
               RealParameter('Inital value for N95 respirators in stockpile UK',0, 78900000),
               RealParameter('Inital value for gowns in stockpile UK',0, 57900000),
               RealParameter('Inital value for gloves in stockpile UK',0, 1079700000),
               RealParameter('Preparation time for delivery PPE',1, 10),
               RealParameter('Delivery time of PPE stockpiling', 1, 21),
               RealParameter('Government budget for PPE',0 , 1),
               RealParameter('Urgentness',0 ,5)]


# In[7]:


from auxiliary import get_last_element
# from LastElement import get_last_element


# In[8]:


results_policies = pd.read_csv('./data/candidate_policy_step4_simplemasks_gloves.csv')
results_policies


# In[9]:


# drop to use as policies
results_policies = results_policies.drop(['Unnamed: 0'], axis=1)

policies_to_evaluate = []

for i, policy in results_policies.iterrows():
    policies_to_evaluate.append(Policy(str(i), **policy.to_dict()))


# In[10]:


# policies_to_evaluate


# In[11]:


Model.outcomes = [ScalarOutcome('Coverage simple masks', variable_name='Total normalized coverage simple masks',
                                            kind=ScalarOutcome.MAXIMIZE, function = get_last_element),
                    TimeSeriesOutcome("Shortage of simple masks per day in UK considering threshold"),
                    TimeSeriesOutcome("Shortage of gloves per day in UK considering threshold"),
                    TimeSeriesOutcome("Total cost for PPE"),
                    TimeSeriesOutcome("Simple masks supply ready to be shipped in UK"),
                    TimeSeriesOutcome("Gloves supply ready to be shipped in UK"),
                    TimeSeriesOutcome("Coverage of simple masks for HCWs considering threshold"),
                    TimeSeriesOutcome("Coverage of gloves for HCWs considering threshold"),
#                     ScalarOutcome('Coverage N95 respirators', variable_name='Total normalized coverage N95 respirators',
#                                             kind=ScalarOutcome.MAXIMIZE,function = get_last_element),
#                     ScalarOutcome('Coverage gowns', variable_name='Total normalized coverage gowns',
#                                             kind=ScalarOutcome.MAXIMIZE, function = get_last_element),
                    ScalarOutcome('Coverage gloves', variable_name='Total normalized coverage gloves',
                                            kind=ScalarOutcome.MAXIMIZE, function = get_last_element)]
#                     ScalarOutcome('Total normalized coverage eye protection', variable_name='Total normalized coverage eye protection',
#                                             kind=ScalarOutcome.MAXIMIZE, function = get_last_element)]


# In[12]:


# define amount of scenarios
n_scenarios = 400

#generation of scenarios, evaluating the policies over the scenarios
with SequentialEvaluator(Model) as evaluator:
    results = evaluator.perform_experiments(n_scenarios,policies_to_evaluate)


# In[ ]:





# In[13]:


save_results(results, './data/results_step3_simple_masks_4.tar.gz')


# In[ ]:




