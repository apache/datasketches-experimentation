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

from DataGenerator import DistributionDataGenerator, Workload
from SyntheticDataGenerators import SyntheticStreamMaker
from QueryGenerator import *
import SketchExperiment
import logging
import Oracle
import random
import numpy as np
import scipy as sp
import copy
import types

import sketches.Sketches as Sketches
logging.getLogger().setLevel(logging.WARNING)

if __name__ == '__main__':     
    
    # query for quantiles 5, 10, ..., 95
    qg1 = ConfigQueryGenerator(
        queries='quantile',
        parameters=np.arange(0.05, 1, 0.05),
        indices=DataGeneratorSeq(length=10),        
    )
    qg2 = ConfigQueryGenerator(
        queries='cdf',
        parameters=np.arange(0.05, 1, 0.05),
        indices=DataGeneratorSeq(length=10),        
    )
    qg1.name = 'quantile'
    qg2.name = 'cdf'
    
    qg = ChainQueryGenerators(generators=[qg1, qg2])    
    
    dg1 = DistributionDataGenerator(length=int(1e5), distribution=sp.stats.beta(1,1), name='Uniform')
    w1 = Workload(data_generator=dg1, query_generator=qg)
    
    dg2 = DistributionDataGenerator(length=int(1e5), distribution=sp.stats.norm(0.5,0.2), name='Normal')
    w2 = Workload(data_generator=dg2, query_generator=qg)
    
    dg3 = SyntheticStreamMaker(n=1e5, order='zoomin')
    w3 = Workload(data_generator=dg3, query_generator=qg1)
    
    oracle = Oracle.QuantileOracle(save_dir = "/tmp/answers", as_json=True, read_cache=True)
    qg.connectDataGenerator(dg2)
    opts = SketchExperiment.ExperimentOptions(
        nparallel=8,
        ndatasets=1,
        nrepetitions=8,
        save_answers=True,
        )
        
    
    e = SketchExperiment.SketchMetaExperiment(workloads=[w1, w2, w3], 
                                      oracle=oracle,
                                      options=opts,
                                      result_file="tmp_quantile_exp_results.csv",
                                      )
    
    #oracle2 = copy.deepcopy(oracle)
#    SketchConfig(sketch=Sketches.KLLSketch, size=range(100,1000,100))
    
    KLL_params = [{'size':s} for s in range(100,1000,200)]
    REQ_params = [{'size':s} for s in range(10,100,20)]
    Tdigest_params = [{'delta':1/s} for s in range(20,200,40)]
    sketches = {'KLL': (Sketches.KLLSketch, KLL_params),
                'REQ': (Sketches.REQSketch, REQ_params),
                'Tdigest': (Sketches.TDigestSketch, Tdigest_params),
                #'Oracle': Sketches.SketchFactory(Sketches.OracleSketch, oracle2),
               }

    e.addSketches(sketches)
    
#    e.prepare()
    e.execute()