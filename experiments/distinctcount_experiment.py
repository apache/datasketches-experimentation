# 
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.
# 

import sys 
import os
pp = os.path.abspath('..')
p = os.path.abspath('.')
sys.path.insert(0, p)
sys.path.insert(0, pp)

import DataGenerator
import SketchExperiment
import logging
import Oracle
import random
import numpy as np
import scipy as sp
import copy
from experiment_utils import makeDict


import sketches.Sketches as Sketches
logging.getLogger().setLevel(logging.WARNING)

if __name__ == '__main__':     
    # A Beta(1,1) is a uniform distribution
    d = sp.stats.beta(1,1)    
    dg = DataGenerator.DistributionDataGenerator(length=int(1e6), distribution=d, name='Uniform')
    qg = DataGenerator.RepeatQueryGenerator(query_type="distinct", repeated_val=0, num_queries=5)
        
    w = DataGenerator.Workload(data_generator=dg, query_generator=qg)
    oracle = Oracle.DistinctStreamOracle(workload=w) # note this oracle assumes the stream consists of all unique items
    opts = SketchExperiment.ExperimentOptions(
        nparallel=8,
        nreorderings=10,
        nrepetitions=10, 
        save_answers=True)
        
    
    e = SketchExperiment.SketchExperiment(workload=w, 
                                          oracle=oracle,
                                          options=opts,
                                          )
    
    sketches = {'Theta': (Sketches.ThetaSketch, makeDict(lg_k=10, p=1.0, seed=0)),
                'HLL': (Sketches.HLLSketch, makeDict(lg_k=10, seed=0)),
               }

    e.addSketches(sketches)
    
    e.prepare()
    e.execute(save_sketch=False)