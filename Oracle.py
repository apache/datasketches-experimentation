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

import pandas as pd
import numpy as np
import copy

import pickle
import tempfile
import os
import json

from bisect import bisect

class Oracle:
    """
    Oracle handles generating the true answers and evaluating/collecting the sketch's answers
    By default, this assumes all query answers are real valued

    The exact method for a problem should be implemented here
    """

    def __init__(self, workload=None, answer_file=None, read_cache=False, save_dir=None, as_json=False, **kwargs):
        """
        Currently, every oracle's init must have a kwargs argument.
        
        This uses kwargs in a less than ideal way to handle different Oracles having
        different signatures in the initialization. This init is called when loading an 
        Oracle's results from the cache.
        """
        self.workload = workload
        self.answers = []
        self.answer_file = answer_file
        self.read_cache = read_cache
        self._prepared = False
        self.save_dir = save_dir
        self.as_json = as_json
        
    def setWorkload(self, workload):
        self._prepared = False
        self.workload = workload
        
    def getID(self):
        return f"Oracle_{self.name}_{self.workload.getID()}"

    def getAnswer(self, qid):
        truth = self.answers[qid]
        return truth
        
    def eval_sketch_answer(self, qid, answer):
        error = self.eval_error(qid, answer)
        return error
    
    #
    # These are the main functions that need to be implemented for each new problem
    #
    def eval_error(self, qid, answer):
        """
        By default, assume errors are real-valued and can be added
        """
        truth = self.answers[qid]
        return answer - truth

    def add(self, x):
        raise Exception

    def query(self, query, parameters):
        raise Exception("Unimplemented")
        
    #
    # Functions to write/read oracle answers to disk
    #
    def getAnswerFile(self):
        prefix = self.getID()

        if self.save_dir is None:
            fd, filename = tempfile.mkstemp(prefix=prefix)
        else:
            filename = f"Answers_{prefix}.json"
            
        self.answer_file = filename

        return filename
                
    def prepareFromCached(self):
        if self.answer_file is None:
            self.answer_file = self.getAnswerFile()

        print("prep from cache oracle", self.answer_file)        
        try:
            if self.as_json:
                with open(self.answer_file, "r") as file:
                    self.answers = json.load(file)                    
            else:
                with open(self.answer_file, "rb") as file:
                    self.answers = pickle.load(file)                                    
                    
            if len(self.answers) > 0:
                return True
        except Exception:
            pass
            
        print("Cannot file {self.answer_file}")
        return False
        
    def writeToCache(self):
        answer_file = self.getAnswerFile()
        self.answer_file = answer_file
        if self.as_json:
            with open(answer_file, "w") as file:
                json.dump(self.answers, file)
        else:
            with open(answer_file, "wb") as file:
                pickle.dump(self.answers, file=file)
        
        # I don't ever close the fd and clean up the file right now XXX
    
    def printAnswers(self):
        print("answers:")
        for a, q in zip(self.answers, self.workload.genQueries()):
            print(q, ":", a)
        
    def prepare(self, **kwargs):
        """
        Iterate through the data and populate the pre-prepared answers
        """
        if self._prepared:
            return 
        
        if self.read_cache:
            self._prepared = self.prepareFromCached()
            if self._prepared:
                print("read from cache")
                return
            
        
        self.workload.prepare()
        
        print(f"reset oracle answers")
        self.answers = []
        query_iter = self.workload.genQueries()
        q = next(query_iter)
        for i, x in enumerate(self.workload.genData()):
            self.add(x)
            while q and i == q.data_idx:
                answer = self.query(q.data_idx, q.query, q.parameters)
                self.answers.append(copy.deepcopy(answer))
                assert(len(self.answers) == q.qid+1)
                q = next(query_iter, None)
                
        self.printAnswers()
        
        self.writeToCache() # note: I should not write to cache if not using parallel processes
        self._prepared = True

    def reset(self, workload):
        self.setWorkload(workload)
        
    def prepareForPickle(self):
        """
        This should remove any large objects
        """
        self.workload.prepareForPickle()

##############################################################################################################

# simple distinct count testing when workload always consists of unique items
class DistinctStreamOracle(Oracle):
    name = 'DistinctStream'
    
    def __init__(self, workload, **kwargs):
        super().__init__(workload, **kwargs)
        self.counter = 0

    def add(self, x):
        self.counter += 1

    def query(self, idx, query, params):
        return idx
    
    def eval_error(self, qid, answer):
        """
        By default, assume errors are real-valued and can be added
        """
        truth = self.answers[qid]        
        return (answer - truth) / truth * 100.
    
    def getCached(self):
        return self
    
    
class TopKOracle(Oracle):
    name = "TopK"
    
    def __init__(self, workload=None, **kwargs):
        super().__init__(workload, **kwargs)
        self.table = {}

    def add(self, x):
        self.table[x] = self.table.get(x, 0) + 1

    # get all top k
    def query(self, idx, query, k):
        s = sorted([(w, x) for x, w in self.table.items()])
        topk = [(x, w) for w, x in reversed(s[-k:])]
        return topk

    def eval_error(self, qid, answer):
        """
        Returns the number of missed items in the result set
        Note that the sketch's answer can include more than k items
        """
        truth = self.answers[qid]
        A = set([x for x, w in truth])
        B = set([x for x, w in answer])
        
        missed = len(A) - len(A.intersection(B))
        return missed
        
    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.table = {}
        
    def prepareForPickle(self):
        super().prepareForPickle()
        self.table = None        

class QuantileOracle(Oracle):
    name = "Quantile"
    
    def __init__(self, workload=None, **kwargs):
        super().__init__(workload, **kwargs)
        self.dat = []
        self.is_sorted = False
        
    def add(self, x):
        self.dat.append(x)
        self.is_sorted = False

    # This sorts and gets the quantile q
    # The quantile is defined to be the lower semicontinuous inverse CDF 
    # That is, it does no interpolation and F^-1(y) = sup {x: F(x) <= y} 
    # where the sup is taken over data points
    def query(self, idx, query, q):
        if not self.is_sorted:
            self.dat.sort()
            self.is_sorted = True

        n = len(self.dat)
        if query == 'quantile':
            rank = int(q * n)            
            return self.dat[rank]
        else:
            i = bisect(self.dat, q)
            return i/n

    def eval_error(self, qid, answer):        
        truth = self.answers[qid]
        return answer-truth
        
    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.dat = []
        self.is_sorted = False

    def prepareForPickle(self):
        super().prepareForPickle()        
        self.dat = None

