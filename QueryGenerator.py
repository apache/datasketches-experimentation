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

from pandas.api.types import is_list_like
import itertools
import heapq
from functools import total_ordering

@total_ordering
class Query:
    def __init__(self, qid, idx, query, parameters, param_idx=0):
        self.data_idx = idx
        self.qid = qid
        self.query = query
        self.parameters = parameters
        self.param_idx = param_idx # used to order queries

    def __iter__(self):
        yield self.data_idx
        yield self.qid
        yield self.query
        yield self.parameters
    
    def info(self):
        return {
            'query_idx': self.data_idx,
            'qid': self.qid,
            'query': self.query,
            'query_parameters': self.parameters,
        }
    
    def __str__(self):
        return f"{self.query} {self.parameters} {self.data_idx}"
    
    def __lt__(self, b):        
        return self.data_idx < b.data_idx
    
    def __eq__(self,b):
        return self.data_idx == b.data_idx
        
class QueryGenerator:
    name = "generic_QG"
    END = 1000000000 
    
    def __init__(self):
        self.prepared = False
        
    def connectDataGenerator(self, data_generator):
        self.data_generator = data_generator

    def getName(self):
        return self.name
            
    def getID(self):
        return self.name
    
class DataGeneratorSeq:
    def __init__(self, data_generator=None, length=None, by=None):
        self.data_generator = data_generator
        self.length = length
        self.by = by
    
    def genSeq(self):
        if self.length is not None:
            n = len(self.data_generator)
            assert(n > 1)
            by = (n-1) / self.length
        else:
            n = QueryGenerator.END
            by = self.by
        
        i = 1
        while i*by < len(self.data_generator):
            yield int(i*by)
            i += 1
        if i*by != n-1:
            yield QueryGenerator.END
            
            
        
class ConfigQueryGenerator(QueryGenerator):
    name = 'config_QG'

    def __init__(self, queries, indices=[QueryGenerator.END], parameters=None):
        self.queries = queries
        self.indices = indices
        self.query_parameters = parameters
    
    def genQueries(self):
        qid = 0
        if isinstance(self.queries, list):
            queries = self.queries
        else:
            queries = [self.queries]
            
        if is_list_like(self.query_parameters) and not isinstance(self.query_parameters, dict):
            query_parameters = self.query_parameters
        else:
            query_parameters = [self.query_parameters]
            
        if isinstance(self.indices, DataGeneratorSeq):
            indices = self.indices.genSeq()
        else:
            indices = self.indices
        
        for q, idx, params in itertools.product(queries, indices, query_parameters):
            yield Query(qid, idx, q, params)
            qid += 1
    
    def connectDataGenerator(self, data_generator):
        self.data_generator = data_generator
        if isinstance(self.indices, DataGeneratorSeq):
            self.indices.data_generator = data_generator
        
class ChainQueryGenerators(QueryGenerator):
    name = 'chained_QG'

    def __init__(self, generators=[]):
        super().__init__()
        self.query_generators = generators
        self.heap = []
        
    def genQueries(self):
        # reassign the qid's and ensure queries are ordered by idx
        iters = [qg.genQueries() for qg in self.query_generators]
            
        for i, it in enumerate(iters):
            q = next(it, None)
            if q is not None:
                self.heap.append((q, i))
            
        heapq.heapify(self.heap)
        qid = 0
        while self.heap:
            q, i = heapq.heappop(self.heap)
            q.qid = qid
            qid += 1
            yield q
            
            nextq = next(iters[i], None)
            if nextq is not None:
                heapq.heappush(self.heap, (nextq, i))
            
    def connectDataGenerator(self, data_generator):
        for qg in self.query_generators:
            qg.connectDataGenerator(data_generator)
        

class TopKQueryGenerator(ConfigQueryGenerator):
    def __init__(self, k_values, num_queries, data_generator=None, **kwargs):
        super().__init__(queries="topk", 
                         indices=DataGeneratorSeq(data_generator, length=num_queries),
                         parameters=k_values)
        