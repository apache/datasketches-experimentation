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
import numpy as np
from random import uniform
from sortedcontainers import SortedList
from collections import namedtuple
Sample = namedtuple('Sample', ['priority', 'weight', 'item'])
from numba import jit

class Sampler:
    def __init__(self, seed=None):
        self.buffer = SortedList()
        self.threshold = float("inf")
        self.processed = 0
        self.seed = seed
        self.rng = random.Random(seed)

    def cdf(self, v, weight):
        F = weight * v
        return min(1.0, F)

    def priority(self, item, weight):
        return self.rng.uniform(0,1./weight)

    def pop(self):
        T, w, x = self.buffer.pop()
        self.threshold = T
        return T, w, x

    def reduceSampleSize(self, k):
        while len(self.buffer) > k:
            self.pop()

    def postAdd(self, item, weight):
        pass

    def add(self, item, weight):
        self.processed += 1
        R = self.priority(item, weight)
        if R < self.threshold:
            self.buffer.add(Sample(R, weight,item))
            self.postAdd(item, weight)

    def getThreshold(self, item, w):
        return self.threshold

    def items(self):
        """
        Generator which returns item, pseudo-inclusion probability
        """
        for R, w, x in self.buffer:
            yield x, self.inc_probability(self.getThreshold(x, w), w)

    def inc_probability(self, x, w):
        return self.cdf(x, w)

##################################################################################################################    
    
def filterSum(sampler, eval = lambda x: x, predicate = lambda x: True):
    s = 0
    var = 0
    for x, pi in sampler.items():
        if predicate(x):
            #pi = sampler.inc_probability(x, w)
            v = eval(x)
            s += v / pi
            var += v*v * (1.0-pi) / (pi*pi)

    return s, var

class BottomKSampler(Sampler):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def postAdd(self, item, weight):
        self.reduceSampleSize(self.k)
  
#################################################################################################################

class SpaceBoundedSampler(Sampler):
    def __init__(self, budget, len=len):
        super().__init__()
        self.budget = budget
        self.size = 0
        self.len = len

    def pop(self):
        T, w, x = super().pop()
        self.size -= self.len(x)

    def compact(self, budget):
        while self.size > budget:
            self.pop()

    def postAdd(self, item, weight):
        self.size += self.len(item)
        self.compact(self.budget)
    
##################################################################################################################

import xxhash

from enum import Enum

BIGVAL64 = (2**64-1)
class MultiStratifiedSampler(Sampler):
    """
    Only compact the sample if the total sample size gets too large
    """
    UNKNOWN = 0
    REMOVE_CANDIDATE = 1
    NOT_CANDIDATE = 2

    def __init__(self, num_objectives, target_size, slack=1.2, seed=None):
        self.target_size = target_size
        self.buffer = SortedList()
        self.thresholds = [float("inf") for i in range(num_objectives)]
        self.composite_threshold = float("inf")
        self.min_size_per_objective = target_size+1
        self.slack = slack
        self.seed = seed
        self.rng = self.Random(seed)

    # use hash based
    def item_rv(self, x):
    #    h = xxhash.xxh64(str(x))
    #    z = h.intdigest() / BIGVAL64
        z = self.rng.uniform(0,1)
        return z 

    def priority(self, U, x, weight):
        return [U / w for w in weight]

    def getThreshold(self, item, w):
        return self.thresholds

    def cdf(self, v, weight):
        p = 0.
        for x, w in zip(v, weight):
            F = min(1.0, w * x)
            p = max(p, F)

        return p

    def pop(self):
        raise Exception

    @classmethod
    def lt(cls, priority, threshold):
        for r, t in zip(priority, threshold):
            if r < t: 
                return True
        return False
   
    def getSizePerObjective(self):
        size_per_objective = [0]*len(self.thresholds)
        for R, w, x in self.buffer:
            #R = self.priority(U,x,w)
            for i, (r, t) in enumerate(zip(R, self.thresholds)):
                if r < t: 
                    size_per_objective[i] += 1
        return size_per_objective

    def compact(self):
        while len(self.buffer) > self.target_size * self.slack:
            if self.min_size_per_objective > self.target_size:
                self.min_size_per_objective = max(self.getSizePerObjective())

            self.min_size_per_objective = int(self.min_size_per_objective / self.slack)
            #print("resize per obj: ", self.min_size_per_objective, len(self.buffer))
            self.compactToSize(self.min_size_per_objective)
            #print(len(self.buffer))

    def getScaledThresholds(self, min_size_per_objective):
        num_objectives = len(self.thresholds)
        scaled_thresholds = []
        for i in range(num_objectives):
            priorities = [R[i] for R, w, x in self.buffer]
            priorities.sort()
            scaled_thresholds.append( priorities[min_size_per_objective+1] )
    
        return scaled_thresholds

    def compactToSize(self, min_size_per_objective):
        self.thresholds = self.getScaledThresholds(min_size_per_objective)
    
        new_buffer = SortedList()
        for s in self.buffer:
            if self.lt(s.priority, self.thresholds):
                new_buffer.add(s)
        self.buffer = new_buffer    
    
    def add(self, item, weight):
        U = self.item_rv(x)
        R = self.priority(U, item, weight)    
        if self.lt(R, self.thresholds):
            self.buffer.add(Sample(R, weight, item))      
            self.compact()

            
from math import sqrt


##################################################################################################################



 #= namedtuple('TopKItem', ['priority', 'item', 'threshold', 'count'])
class TopKItem:
    def __init__(self, priority, item, weight, threshold, count):
        self.priority = priority
        self.item = item
        self.weight = weight
        self.threshold = threshold
        self.count = count  

    def __lt__(self, other):
        return self.priority < other.priority


    # 1/min(1, threshold) is a hack
    def nhat(self, f=lambda x: x):
        #return f(self.item) * 
        return (1/min(1, self.threshold) + self.count)

    def __str__(self):
        return f"{self.item} {self.weight} {self.count}"

class TopKSampler(Sampler):
    def __init__(self, topk, maxsize, seed=None):
        self.topk = topk
        self.maxsize = maxsize
        self.buffer = [] #SortedList()
        self.heavy_set = set()
        self.item_dict = {}
        self.threshold = float("inf")
        self.processed = 0
        self.seed = seed
        self.rng = random.Random(seed)

    def size(self):
        return len(self.item_dict)

    def nhat_infreq(self):
        return 1. / self.threshold
  
    def is_infreq(self, item, mingap=0):
        y = self.item_dict[item] 
        if y.count == 0 or y.nhat() - mingap < self.nhat_infreq():
            return True
        return False

    def getNumHeavy(self):
        return sum([not self.is_infreq(x, 0.1*self.nhat_infreq()) for x in self.heavy_set])

    def items(self):
        """
        Generator which returns item, pseudo-inclusion probability
        """
        for topk_item in self.buffer:
            R = topk_item.priority 
            w = topk_item.weight
            x = topk_item.item
            yield x, self.inc_probability(self.getThreshold(x, w), w)

    def getTotal(self):
        return sum([x.nhat() for x in self.buffer])

    def compact(self):
        if self.getNumHeavy() <= self.topk and self.size() <= self.maxsize:
            return
    
        self.buffer.sort()

        #print("compact", self.size(), self.getNumHeavy(), self.threshold, self.getTotal(), self.processed)
        while self.size() > self.maxsize or self.getNumHeavy() > self.topk:
            topk_item = self.buffer.pop()
            del self.item_dict[topk_item.item]
            if topk_item.item in self.heavy_set:
                self.heavy_set.remove(topk_item.item)
            self.threshold = min(self.threshold, topk_item.priority) # shouldn't need take min
      

    def add(self, item, weight):
        self.processed += 1
        if item in self.item_dict:
            topk_item = self.item_dict[item]
            #print("initinc", topk_item.priority, topk_item.count, item, weight)
            nhat = topk_item.nhat()
            topk_item.priority *= nhat / (nhat+1)
            topk_item.count += 1
            #print("inc", topk_item.priority, topk_item.count, item, weight)
            if not self.is_infreq(item):
                self.heavy_set.add(item)
            return

        R = self.priority(item, weight)
        #print("add", item, R)
        if R < self.threshold:
            entry = TopKItem(R, item, weight, self.threshold, 0)
            self.buffer.append(entry)
            self.item_dict[item] = entry
            self.compact()
    

##################################################################################################################
  



