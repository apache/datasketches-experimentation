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


import random
import scipy
import scipy.stats
import numpy as np
import itertools
import pandas as pd 
import os

from DataGenerator import DataGenerator, Workload, DistributionDataGenerator
    
from StreamMaker import StreamMaker

class SyntheticStreamMaker(DataGenerator):
    name = 'StreamMaker'
    valid_orders = ['sorted','reversed','zoomin','zoomout','sqrt','random','adv','clustered', 'clustered-zoomin']
    
    
    def __init__(self, n, order, p=1000, g=0, s=1, seed=None):
        self.stream_maker = StreamMaker(seed)
        self.n = int(n)
        self.order = order
        self.p = p
        self.g = g
        self.s = s
        self.seed = seed
        
    def __len__(self):
        return self.n
    
    def genData(self):
        for x in self.stream_maker.make(self.n, self.order, self.p, self.g, self.s):
            yield x
    
    def getName(self):
        return f"{self.name}:{self.order}"
        
    def reset(self, seed=None):
        self.seed = seed
        self.stream_maker.rng.seed(seed)
        
##########################################################################################

from random import randint

class PitmanYorDataGenerator(DataGenerator):
    name = "Two-parameter Poisson-Dirichlet"

    def __init__(self, length, alpha, beta, *args, **kwargs):
        super().__init__(**kwargs)  # need to cooperate with other classes for multiple inheritance
        self.size = length
        self.alpha = alpha
        self.beta = beta
        self.atoms = []
        self.roots = set()
        self.nclusters = 0
        
        
    def __len__(self):
        return self.size

    def reset(self, seed):
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.atoms = []
        self.roots = set()
        self.nclusters = 0
        
        
    def genData(self):
        for i in range(self.size):
            U = self.rng.uniform(0, i + self.alpha)
            do_split = self.rng.uniform(0, 1) < self.beta
            parent = int(U)
            if U >= i or (parent in self.roots and do_split):
                self.nclusters += 1
                self.atoms.append(self.nclusters)
                self.roots.add(i)
                yield self.nclusters
            else:
                self.atoms.append(self.atoms[parent])
                yield self.atoms[parent]


# vector valued data generators      
class BinaryVecDataGenerator(DataGenerator):
    """
    takes a scipy.stats distribution
    and assigns it its own rng with a specified seed
    """
    def __init__(self, length, distribution, name, seed=0, dim=1, *args, **kwargs):
        super().__init__(**kwargs)  # need to cooperate with other classes for multiple inheritance
        self.size = length
        self.distribution = distribution
        self.seed = seed
        self.dim = dim
        self.name = name
        
    def __len__(self):
        return self.size

    def prepareData(self):
        pass
    
    def genData(self):
        d = self.distribution
        d.random_state = np.random.default_rng(seed=self.seed)
        np_rng = np.random.RandomState(seed=self.seed)
        for i in range(self.size):
            x = d.rvs(1)[0]
            while x > self.dim:
                x = d.rvs(1)[0]
            pi = np_rng.permutation(self.dim)
            idx = pi[:x]
            z = np.zeros(self.dim)
            z[idx] = 1.0
            yield z
        
class DistributionDataGeneratorWithDupes(DistributionDataGenerator):
    """
    takes a scipy.stats distribution
    and assigns it its own rng with a specified seed
    """
    def __init__(self, dupes=0, **kwargs):
        super().__init__(**kwargs)  # need to cooperate with other classes for multiple inheritance
        self.dupes=dupes
        assert(dupes < self.dim)
    
    def genData(self):
        for x in itertools.islice(self.genDistributionSequence(dim=self.dim),self.size):
            x[:self.dupes] = x[0]
            yield x

############################################################################################################
from QueryGenerator import *        

class PitmanYorWorkload(Workload):
    name = "Pitman-Yor"
    
    def __init__(self, length, alpha, beta, k, num_queries, **kwargs):
        super().__init__(**kwargs)
        self.data_generator = PitmanYorDataGenerator(length=length, alpha=alpha, beta=beta)
        self.query_generator = TopKQueryGenerator(k=k, num_queries=num_queries)

class RetailTopKWorkload(Workload):
    name = "Retail"
    
    def __init__(self, k, num_queries, **kwargs):
        super().__init__(**kwargs)
        self.data_generator = FileDataGenerator(filename="/Users/dting/research/data/heavyhitters/retail.dat")
        self.query_generator = TopKQueryGenerator(k_values=k, num_queries=num_queries)

class WebdocsTopKWorkload(Workload):
    name = "Webdocs"
    
    def __init__(self, k, num_queries, **kwargs):
        super().__init__(**kwargs)
        self.data_generator = FileDataGenerator(filename="/Users/dting/research/data/heavyhitters/webdocs.dat")
        self.query_generator = TopKQueryGenerator(k=k, num_queries=num_queries)

########################################################################################################################

