

if __name__ == '__main__':

    #importing needed packages
    import platypus
    import numpy as np

    from ema_workbench import (RealParameter, ScalarOutcome, IntegerParameter, ema_logging,
                                SequentialEvaluator )

    from ema_workbench.connectors.vensim import VensimModel
    from ema_workbench.em_framework.parameters import Scenario
    from ema_workbench.em_framework.optimization import (HyperVolume,
                                                        EpsilonProgress)

    ema_logging.log_to_stderr(ema_logging.INFO )
        
    from os import getcwd
    wd = getcwd()
    #import model
    Model = VensimModel('Model', wd= wd, model_file= 'Model/Infection Model Testing 02032022 Test.vpmx')

    #referene scenario - based on worst case for simple masks and gloves 
    ref_scenario = {'Average time to approve and develop PPE': 79.17634288425248,
    'Average time to approve and develop products': 37.97153705605813,
    'Base capacity innovation': 13.356749708557222,
    'Base production capacity direct tender ventilator': 394.3537396080108,
    'Base raw material N95 respirators capacity direct tender': 33629.066422013246,
    'Base raw material N95 respirators capacity innovation PPE': 16344.493630854366,
    'Base raw material eye protection capacity direct tender': 139406.76631043447,
    'Base raw material eye protection capacity innovation PPE': 28182.108290345517,
    'Base raw material gloves capacity direct tender': 826104.1995889662,
    'Base raw material gloves capacity innovation PPE': 153372.35029531104,
    'Base raw material gowns capacity direct tender': 237972.05078569372,
    'Base raw material gowns capacity innovation PPE': 4090.0864981166606,
    'Base raw material procurement N95 respirators domestic production': 48155.0508022354,
    'Base raw material procurement N95 respirators worldwide': 725656.4453947457,
    'Base raw material procurement eye protection domestic production': 45271.43493320816,
    'Base raw material procurement eye protection worldwide': 2378172.6117345896,
    'Base raw material procurement gloves domestic production': 796839.0112000088,
    'Base raw material procurement gloves worldwide': 1239373382.8546948,
    'Base raw material procurement gowns domestic production': 90085.32054779386,
    'Base raw material procurement gowns worldwide': 2120311.4117694693,
    'Base raw material procurement simple masks domestic production': 579399.6158656235,
    'Base raw material procurement simple masks worldwide': 27447779.5342482,
    'Base raw material simple masks capacity direct tender': 146227.26745205306,
    'Base raw material simple masks capacity innovation PPE': 22861.93587837413,
    'Base raw material ventilator production worldwide': 1274.7314237455314,
    'Check up time': 17.716237090190734,
    'Delay domestic production PPE': 56.39838419288782,
    'Delay domestic production setup ventilator': 59.17457300505236,
    'Delayed shipment time': 145.1370852542564,
    'Delivery time for available ventilators': 15.940312509405821,
    'Delivery time of ventilators stockpiling': 16.191826477893677,
    'Direct tender set up time PPE': 33.342145631389435,
    'Direct tender set up time ventilator': 16.254004862287154,
    'Eye protection changes per patient per day ICU': 8.475562618525498,
    'Eye protection changes per patient per day non ICU': 1.9874194706842312,
    'Gloves changes per patient per day ICU': 124.7320479216878,
    'Gloves changes per patient per day Non ICU': 86.33554960924845,
    'Government budget for PPE': 0.5577560769229629,
    'Gowns changes per patient per day ICU': 24.990106729713855,
    'Gowns changes per patient per day non ICU': 26.492063452733408,
    'Infectivity': 0.11845528183212853,
    'Initial ventilators in stockpile': 7940.748573010515,
    'Maximum days in backlog before increase in procu capacity': 28.824017371133017,
    'Maximum days in backlog before increase in procu capacity vent': 24.5383846907132,
    'Maximum days in backlog before increase in prod capacity': 3.6416011011757714,
    'Maximum days in backlog before increase in prod capacity vent': 15.833304254874125,
    'Maximum increase in procurement capacity PPE': 16.490625978449998,
    'Maximum increase in procurement capacity vent ww': 2.0426228682168555,
    'Maximum increase in production capacity PPE': 12.831131700546438,
    'Maximum increase in production capacity vent worldwide': 12.361240260704772,
    'Maximum prod direct tender PPE': 11.891325229687144,
    'Maximum prod direct tender vent': 1.5275333521628225,
    'Maximum transportation time': 2.1131258052027446,
    'Maximum transportation time PPE procurement world market': 5.607269233309754,
    'N95 respirators changes per patient per day ICU': 5.354854928699174,
    'N95 respirators changes per patient per day non ICU': 3.4543637617713205,
    'Normal shipment time': 42.873841354948894,
    'Potentially available ventilators': 2217.597905689291,
    'Preparation shipment PPE production worldwide': 1.5238495132079382,
    'Preparation shipment production worldwide': 5.4579731923913135,
    'Preparation time for delivery': 7.223256101679287,
    'Production capacity domestic production ventilator': 192.8896807306737,
    'Production time Innovation PPE': 2.337467221156108,
    'Production time domestic production': 2.5709756022484043,
    'Production time domestic production PPE': 2.732662470762583,
    'Production time innovation ': 1.4544123797167443,
    'Production time ventilator production worldwide': 6.737621705004715,
    'Production time ventilators direct tender': 6.0420726711714,
    'Purchasing power UK as share of GDP per person': 0.15083164876210287,
    'Raw material domestic production ventilator': 346.7370352423588,
    'Reach ': 1348.0003689176685,
    'Reach PPE': 120.014364892043,
    'Reduction export PPE': 0.07772894604546987,
    'Reduction export ventilator': 0.6693731870188007,
    'Set up time procurement PPE worldwide': 36.8352165533947,
    'Setting up innovation process': 21.95151774504818,
    'Setting up innovation process PPE': 16.418902136988756,
    'Share of ICU used for other patients': 0.7067724931390581,
    'Share of PPE ready for previous order': 0.3076629713039312,
    'Share of actionable innovations': 0.04912237260330855,
    'Share of faulty PPE': 0.4840464260542451,
    'Share of faulty products': 0.0829277819777853,
    'Share of hospital beds used for other patients': 0.0695108601818353,
    'Share of vent ready for previous order': 0.479581662274419,
    'Share of ventilators available and fitting': 0.8481290155278203,
    'Shipment time direct tender': 90.14936505291023,
    'Shipment time direct tender PPE': 54.88533558439988,
    'Shipment time domestic production': 9.494166055911377,
    'Shipment time domestic production PPE': 19.041358576392753,
    'Shipment time innovation ': 15.942825204952685,
    'Shipment time innovation PPE': 11.662874582828096,
    'Shipment time procurement from world market': 68.00809412768079,
    'Simple mask changes per patient per day ICU': 10.629076528169367,
    'Simple mask changes per patient per day non ICU': 14.989206937101775,
    'Switch quarantine old people': 1.0,
    'Threshold for export restriction PPE': 23011903.16688233,
    'Threshold to start direct tender PPE': 16.060609826973902,
    'Time horizon for forecast': 16.635521602569266,
    'Time to reach max direct tender': 51.68237297890237,
    'Time to reach maximum procurement capacity PPE dom production': 86.25291859813257,
    'Time to reach maximum procurement capacity PPE innovation': 43.07276380281855,
    'Time to reach maximum procurement capacity PPE worldwide': 59.539916375670785,
    'Time to reach maximum procurement capacity vent dom production': 10.631298865164382,
    'Time to reach maximum procurement capacity vent innovation': 38.16009038163432,
    'Time to reach maximum procurement capacity vent worldwide': 284.3385117220554,
    'Time to reach maximum production capacity PPE dom production': 75.14400406209562,
    'Time to reach maximum production capacity PPE innovation': 49.4430841683793,
    'Time to reach maximum production capacity PPE worldwide': 121.44226488479576,
    'Time to reach maximum production capacity vent dom production': 19.851051287047575,
    'Time to reach maximum production capacity vent innovation': 49.123038560476225,
    'Time to reach maximum production capacity vent worldwide': 387.6252556175428,
    'Transportation time PPE innovation': 9.903166757574864,
    'Transportation time direct tender PPE': 2.725599543996106,
    'Transportation time direct tender ventilator': 1.3464897944457606,
    'Transportation time domestic production': 9.285655757869963,
    'Transportation time domestic production PPE': 1.8410980603397369,
    'Transportation time ventilator innovation': 3.2253464327945323,
    'change in transportation time': 36.33822063500757,
    'change in transportation time PPE': 21.207654065010807}


    ref_scenario = Scenario('ref_scenario', **ref_scenario)

    #define uncertainties
    #uncertainties demand 
    Model.uncertainties = [
    RealParameter('Gloves changes per patient per day ICU',85,250),
    RealParameter('Gloves changes per patient per day Non ICU',40 ,120),
    RealParameter('Gowns changes per patient per day ICU',10, 30),
    RealParameter('Gowns changes per patient per day non ICU',10, 30),
    RealParameter('Simple mask changes per patient per day ICU',5, 15),
    RealParameter('Simple mask changes per patient per day non ICU',5, 15),
    RealParameter('N95 respirators changes per patient per day ICU',2, 6),
    RealParameter('N95 respirators changes per patient per day non ICU',1.3 , 3.9),
    RealParameter('Eye protection changes per patient per day ICU',3, 9),
    RealParameter('Eye protection changes per patient per day non ICU',1.3 , 3.9),

    # uncertainties forecast type
    RealParameter('Time horizon for forecast',7 , 30),

    # uncertainties infection model
    RealParameter('Infectivity',0.02, 0.2),

    # uncertainties stockpiling PPE

    # domestic production PPE
    RealParameter('Transportation time domestic production PPE',1, 14),
    RealParameter('Base raw material procurement eye protection domestic production',20000,240000),
    RealParameter('Base raw material procurement simple masks domestic production',20000,900000),
    RealParameter('Base raw material procurement N95 respirators domestic production',5000,50000),
    RealParameter('Base raw material procurement gowns domestic production',20000,240000),
    RealParameter('Base raw material procurement gloves domestic production',144000,1440000),
    RealParameter('Shipment time domestic production PPE',1,21),
    RealParameter('Production time domestic production PPE',1,10),
    RealParameter('Time to reach maximum production capacity PPE dom production',5, 90),
    RealParameter('Time to reach maximum procurement capacity PPE dom production',5, 90), 
                        

    # direct tender PPE
    RealParameter('Transportation time direct tender PPE',1,21),
    RealParameter('Threshold to start direct tender PPE',1,21),
    RealParameter('Time to reach max direct tender',1,90),
    RealParameter('Maximum prod direct tender PPE',1,12),
    RealParameter('Base raw material eye protection capacity direct tender',20000, 400000),
    RealParameter('Base raw material simple masks capacity direct tender',50000, 2000000),
    RealParameter('Base raw material N95 respirators capacity direct tender',10000, 100000),
    RealParameter('Base raw material gowns capacity direct tender',20000, 400000),
    RealParameter('Base raw material gloves capacity direct tender',800000, 50000000),
    RealParameter('Shipment time direct tender PPE',14, 120),
    RealParameter('Share of faulty PPE',0, 0.5),

                        
    #innovation PPE

    RealParameter('Reach PPE',0, 300),
    RealParameter('Production time Innovation PPE',1, 7),
    RealParameter('Transportation time PPE innovation',3.5, 21),
    RealParameter('Shipment time innovation PPE',3.5, 14),
    RealParameter('Base raw material eye protection capacity innovation PPE',500, 30000),
    RealParameter('Base raw material simple masks capacity innovation PPE',3400, 34000 ),
    RealParameter('Base raw material N95 respirators capacity innovation PPE',3400, 25200),
    RealParameter('Base raw material gowns capacity innovation PPE',500, 30000 ),
    RealParameter('Base raw material gloves capacity innovation PPE',5000, 160000 ),
    RealParameter('Average time to approve and develop PPE',15, 120),
    RealParameter('Time to reach maximum production capacity PPE innovation',5, 90),
    RealParameter('Time to reach maximum procurement capacity PPE innovation',5, 90),                     


    # procurement worldwide PPE
    RealParameter('Base raw material procurement eye protection worldwide',1000000,8000000),
    RealParameter('Base raw material procurement simple masks worldwide',10000000,60000000),
    RealParameter('Base raw material procurement N95 respirators worldwide',120000,1200000),
    RealParameter('Base raw material procurement gowns worldwide',1000000,8000000),
    RealParameter('Base raw material procurement gloves worldwide',200000000,1400000000),
    RealParameter('Preparation shipment PPE production worldwide',1, 10),
    RealParameter('Threshold for export restriction PPE',1000000, 100000000),
    RealParameter('Delayed shipment time',30, 360),
    RealParameter('Normal shipment time',7, 45),
    RealParameter('Reduction export PPE',0, 1),
    
    RealParameter('Maximum increase in procurement capacity PPE',1, 20),
    RealParameter('Time to reach maximum procurement capacity PPE worldwide',14, 210),
    RealParameter('Maximum days in backlog before increase in procu capacity',1, 45),       
    RealParameter('Maximum increase in production capacity PPE',1, 20),
    RealParameter('Time to reach maximum production capacity PPE worldwide',14, 210),
    RealParameter('Maximum days in backlog before increase in prod capacity',1, 20),   
    RealParameter('Share of PPE ready for previous order',0.2, 1),
    RealParameter('Maximum transportation time PPE procurement world market',1, 20),
    RealParameter('change in transportation time PPE',7, 60)]

    #define model levers
    Model.levers = [IntegerParameter('Switch procurement world market PPE',0,1),
                IntegerParameter('Switch direct tender PPE',0,1),
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
#                 RealParameter('Time to check products',1, 5),
#                 RealParameter('Shipment time to hospitals',1, 10),

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
    
    #import function get last element
    from LastElement import get_last_element

    #define model outcomes
    Model.outcomes = [ScalarOutcome('Coverage simple masks', variable_name='Total normalized coverage simple masks',
                                            kind=ScalarOutcome.MAXIMIZE, function = get_last_element),
                    ScalarOutcome('Coverage N95 respirators', variable_name='Total normalized coverage N95 respirators',
                                            kind=ScalarOutcome.MAXIMIZE,function = get_last_element),
                    ScalarOutcome('Coverage gowns', variable_name='Total normalized coverage gowns',
                                            kind=ScalarOutcome.MAXIMIZE, function = get_last_element),
                    ScalarOutcome('Coverage gloves', variable_name='Total normalized coverage gloves',
                                            kind=ScalarOutcome.MAXIMIZE, function = get_last_element),
                    ScalarOutcome('Total normalized coverage eye protection', variable_name='Total normalized coverage eye protection',
                                            kind=ScalarOutcome.MAXIMIZE, function = get_last_element)]

    #defining convergence metric
    convergence_metrics =[EpsilonProgress()]

    #running directed search with sequential evaluator
    with SequentialEvaluator(Model) as evaluator:
        results, convergence = evaluator.optimize(nfe=5000, searchover='levers', convergence=convergence_metrics,
                                        epsilons=[0.05,] * len(Model.outcomes) , Scenario=ref_scenario, logging_freq = 1)
        
        #saving results and convergence metric
    results.to_excel('./data/5000_results_directed_search_gloves_simple_masks.xlsx')
    convergence.to_excel('./data/5000_convergence_gloves_simple_masks.xlsx')