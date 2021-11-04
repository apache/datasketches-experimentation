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

from DataGenerator import PitmanYorDataGenerator, FileDataGenerator, Workload
from QueryGenerator import TopKQueryGenerator
import SketchExperiment
import logging
import Oracle
import random
import numpy as np
import scipy as sp
import copy
import types
import itertools 

import sketches.Sketches as Sketches
logging.getLogger().setLevel(logging.WARNING)

if __name__ == '__main__':     
    
    k=[3,5,7] + list(range(10,100,10))
    sizes = [100,500, 1000]
    dg = PitmanYorDataGenerator(length=int(1e5), alpha=1,beta=0.5)
    qg = TopKQueryGenerator(k_values=k, num_queries=10)
    w  = Workload(dg, qg)
    oracle = Oracle.TopKOracle(workload=w)
    
    
    opts = SketchExperiment.ExperimentOptions(
        nparallel=8,
        ndatasets=10,
        nrepetitions=1,
        save_answers=True,        
    )
    
    for w in workloads:
        e = SketchExperiment.SketchExperiment(workload=w, 
                                              oracle=oracle,
                                              options=opts,
                                              result_file="topk_exp_results.csv"
                                              )
    
    
    sketches = {'FrequentItems': (Sketches.FrequentItemsSketch, [{"k":v, "maxsize":sz, "cast_type":str} for v,sz in itertools.product(k, sizes)]),
                'TopKSampler': (Sketches.TopKSamplerSketch, [{"k":v, "maxsize":1000} for v in k]),
               }

    e.addSketches(sketches)
    
    e.prepare()
    e.execute(save_sketch=False)