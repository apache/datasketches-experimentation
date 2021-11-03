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


class DataGenerator:
    rng = None
    data = None  # iteratable providing the data stream
    queries = None  # iterable providing (index, query) tuples where answer is the desired answer after processing index data items
    name = "GenericData"
    
    MATERIALIZE = False
    
    def __init__(self, *args, **kwargs):
        self.seed = kwargs.get('seed', 0)
        self.current_seed = None
        self.rng = random.Random(self.seed)
        self.prepared = False
        
    def prepareData(self, size=None, *args, **kwargs):
            pass

    def genData(self):
        raise Exception

    def getName(self):
        return self.name
    
    def getID(self):
        return f"{self.getName()}_{str(self.seed)}"
    
    def __len__(self):
        """
        Return the length of the data stream
        """
        raise Exception

#   These should be filled out in the future to serialize workloads
#   This would allow tests to be run/written in another language but share the same workload/answers/evaluation
#
#     def writeToCache(self):
#         pass
    
#     def prepareFromCache(self):
#         pass
    
#     def getCached(self):
#         return self

    def prepareForPickle(self):
        """
        This should remove any large objects
        """
        pass
    
    def reset(self, seed):
        """
        Update the seed on the data generator
        Typically this generates a new sequence for synthetic data and
        permutes the input order for data from a file.
        """
        self.seed = seed
        self.rng = random.Random(self.seed)


class Workload:
    name = "BaseWorkload"
    data_generator = None
    query_generator = None

    def __init__(self, data_generator = None, query_generator=None, **kwargs):
        self.prepared = False
        self.data_generator = data_generator
        self.query_generator = query_generator
        
    def getName(self, data=True, query=False):
        data_name = self.data_generator.getName()
        query_name = self.query_generator.getName()
        if data and query:
            return f"Data_{data_name}_Queries_{query_name}"
        elif data:
            return data_name
        elif query:
            return query_name
        else:
            return self.name
            
    def getID(self):
        data_name = self.data_generator.getID()
        query_name = self.query_generator.getID()
        return f"Data_{data_name}_Queries_{query_name}"
        
    def genData(self):
        return self.data_generator.genData()
    
    def __len__(self):
        return len(self.data_generator)
    
    def genQueries(self):
        return self.query_generator.genQueries()    

    def prepareData(self):
        if self.data_generator is None:
            raise Exception
            
        self.data_generator.prepareData()

    def prepare(self):
        if self.prepared:
            return
        
        self.prepareData()
        self.query_generator.connectDataGenerator(self.data_generator)
        
    def reset(self, seed=None):
        self.data_generator.reset(seed=seed)
        self.prepared = False
       
    def info(self):
        info = {'workload': self.getName(),
                'data_seed': self.data_generator.seed,
               }
        return info
    
    def prepareForPickle(self):
        self.data_generator.prepareForPickle()
    
    
##########################################################################################

from random import randint

class DistributionDataGenerator(DataGenerator):
    """
    takes a scipy.stats distribution and generates data using it
    In particular, the data generator is assigned its own rng with a specified seed
    """
    def __init__(self, length, distribution, name, seed=0, params={}, dim=1, *args, **kwargs):
        super().__init__(**kwargs)  # need to cooperate with other classes for multiple inheritance
        self.size = length
        self.distribution = distribution
        self.seed = seed
        self.params = params
        self.dim = dim
        self.name = name
        self.chunksize = 1000
        
    def __len__(self):
        return self.size

    def prepareData(self):
        pass
    
    def genDistributionSequence(self, dim=1):
        d = self.distribution  # (**self.params)
        d.random_state = np.random.default_rng(seed=self.seed)
        chunksize = self.chunksize
        while True:
            self.buffer = d.rvs(chunksize*dim)
            if dim==1:
                for x in self.buffer:
                    yield x
            else:
                for i in range(chunksize):                        
                    yield self.buffer[i:(i+dim)]
        
        self.buffer = None
        
    def genData(self):
        return itertools.islice(self.genDistributionSequence(dim=self.dim), self.size)
        
    def prepareForPickle(self):
        self.buffer = None

class FileDataGenerator(DataGenerator):
    """
    Takes a file of tokens and plays it back as a stream.     
    """
    def __init__(self, filename=None, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.data = None
        self.name = os.path.basename(filename)
        
    def prepareData(self, **kwargs):
        if self.prepared:
            return
        
        self.data = []
        with open(self.filename, 'r') as f:
            for line in f:
                tokens = line.rstrip().split()
                self.data.extend(tokens)
                
        self.data = pd.Series(self.data)
        self.prepared = True
        print(f"finished reading {self.filename}")
        
    def __len__(self):
        if self.data is None:
            self.prepareData()
        
        return len(self.data)
    
    def genData(self):  
        # don't support permutations yet.
        return self.data.sample(frac=1, random_state=self.seed)
     
    def prepareForPickle(self):
        self.data = None
        self.prepared = False    

class CSVFileDataGenerator(FileDataGenerator):
    """
    Take a csv file and turns the specified column into a permuted stream
    """
    
    MATERIALIZE = True
    
    def __init__(self, filename=None, column=None, filetype='csv', *args, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.filetype = filetype
        self.column = column
        self.name = filename
        self.data = None
        self.cache_file = None
        
    def prepareData(self, **kwargs):
            
        if self.prepared:
            return
        
        try:
            df = pd.read_csv(self.filename) #, self.filetype)
            self.data = df[self.column]
        except Exception as e:
            print(e)
        
        self.prepared = True


# Given a distribution d, returns
# a random binary vector with X ~ d non-zero entries
class BinaryVecDataGenerator(DataGenerator):
    """
    takes a scipy.stats distribution
    and assigns it its own rng with a specified seed
    """
    def __init__(self, length, distribution, name, seed=0, params={}, dim=1, *args, **kwargs):
        super().__init__(**kwargs)  # need to cooperate with other classes for multiple inheritance
        self.size = length
        self.distribution = distribution
        self.seed = seed
        self.params = params
        self.dim = dim
        self.name = name
        
    def __len__(self):
        return self.size

    def prepareData(self):
        pass
    
    def genData(self):
        d = self.distribution  # (**self.params)
        d.random_state = np.random.default_rng(seed=self.seed)
        np_rng = np.random.RandomState(seed=self.seed)
        for i in range(self.size):
            x = d.rvs(1)[0]
            while x > self.dim:
                x = d.rvs(1)[0]
            pi = np_rng.permutation(self.dim)
            idx = pi[:x]
            z = np.zeros(self.dim) + 0.1
            z[idx] = 1.0
            yield z
        