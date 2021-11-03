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
import sketches.Sampler as Sampler

from collections.abc import Mapping, Container
from sys import getsizeof
 
import numpy as np
import random

def prefixDict(d, prefix=""):
    if not prefix:
        return d
    return dict( (f"{prefix}{k}", v) for k, v in d.items())
    
class Sketch:
    name = "BaseSketch"
    is_random = True
    params = None
    
    def add(self, x, **kwargs):
        raise Exception
    
    def query(self, q=None, parameters=None):
        raise Exception
    
    def getSize(self):
        return 0
    
    def getSizeBytes(self):
        # should be a self reported measurement of size        
        return sys.getsizeof(self)
        
    def info(self, prefix="sketch_"):
        """
        Put stats that should go into dataframe here
        """
        stats = {
            "name": self.name,
            "size": self.getSize()            
        }
        
        return prefixDict(stats, prefix)
    

class SketchFactory:
    """
    This factory method helps the sketch creation function get pickled properly
    
    It takes 1) a sketch class and 2) arguments for creating a sketch and binds the arguments to yield
    a function that creates a sketch with the given arguments
    
    It avoids problems with using lambda function to bind arguments where classes referenced inside the 
    lambda function are not serialized and have no external reference when deserialized
    """
    def __init__(self, sketch_class, seed, *args, **kwargs):
        self.sketch_class = sketch_class
        self.args = args
        self.kwargs = kwargs
        self.seed = seed

    def __call__(self, seed=None):
        return self.sketch_class(*self.args, **self.kwargs, seed=seed)
        
    
##########################################################################################

# for testing
class NoiseSketch(Sketch):
    name = "NoiseSketch"
    
    def add(self, x, **kwargs):
        pass
    
    def query(self, q, parameters=None):
        return uniform(0,1)


class TopKSamplerSketch(Sketch):
    def __init__(self, k, maxsize, seed=None):
        self.topk_sampler = Sampler.TopKSampler(k, maxsize, seed=seed)
        self.k = k
        self.maxsize = maxsize
        
    def add(self,x):
        self.topk_sampler.add(x, 1)
    
    def query(self, q=None, parameters=None):
        """
        return topK item, cnt pairs
        """
        arr = [(topk_item.nhat(), topk_item.item) for topk_item in self.topk_sampler.buffer]
        arr.sort(reverse = True)
        return [(x,nhat) for nhat, x in arr[:min(self.k,len(arr))]]

    def getSizeBytes(self):
        return sum([sys.getsizeof(x) for x in self.topk_sampler.buffer])

    def getSize(self):
        return len(self.topk_sampler.buffer)
    
    def info(self, prefix="sketch_"):
        info = super().info(prefix=None)
        info["seed"] = self.topk_sampler.seed
        info['maxsize'] = self.maxsize
        info['k'] = self.k
        
        return prefixDict(info, prefix=prefix)
    

########################################################################
from datasketches import frequent_strings_sketch, frequent_items_error_type
from math import log2,ceil

class FrequentItemsSketch(Sketch):
    is_random = False   # deterministic sketch
    cast_type = str
    
    def __init__(self, k, maxsize, cast_type=str, seed=None):
        self.k = k
        self.maxsize = maxsize
        v = int(ceil(log2(maxsize)))
        self.sketch = frequent_strings_sketch(v)
        self.cast_type= cast_type
                
    def add(self, x, w=1):
        self.sketch.update(str(x), w)
        
    def query(self, q=None, parameters=None):
        items = self.sketch.get_frequent_items(frequent_items_error_type.NO_FALSE_NEGATIVES)
        result = [(self.cast_type(x), w) for x, lb, ub, w in items]        
        return result[:self.k]
    
    def getSizeBytes(self):
        return self.sketch.get_serialized_size_bytes()
        
    # Not sure this is right
    def getSize(self):
        return self.sketch.get_num_active_items()
    
    def info(self, prefix="sketch_"):
        info = super().info(prefix=None)
        info['bytes'] = self.getSizeBytes()
        info['maxsize'] = self.maxsize
        info['k'] = self.k
        
        return prefixDict(info, prefix=prefix)
        
class OracleSketch(Sketch):
    """
    This turns an oracle into an "exact" sketch
    This would be more useful if the sketch sizes could be calculated here
    """
    def __init__(self, oracle, seed=None):
        self.oracle = oracle
        
    def add(self, x, w=1):
        self.oracle.add(x)
    
    def query(self, q=None, parameters=None):
        return self.oracle.query(None, q, parameters)
    
    

#####################################################
# quantile sketches

from datasketches import kll_floats_sketch, req_floats_sketch
from tdigest import TDigest

class KLLSketch(Sketch):
    name = 'KLL'
    
    def __init__(self, size=200, seed=None):
#         if seed is not None:
#             print("KLL sketch does not support seeding rng")
        self.sketch = kll_floats_sketch(size)
        
    def add(self, x):
        self.sketch.update(x)
        
    def query(self, q=None, parameters=None):
        if q == 'quantile':
            return self.sketch.get_quantile(parameters)
        else:
            return self.sketch.get_cdf([parameters])[0]
    
    def getSizeBytes(self):
        return self.sketch.get_num_retained()*8
    
    def getSize(self):
        return self.sketch.get_num_retained()

    def info(self, prefix="sketch_"):
        info = super().info(prefix=None)
        info['bytes'] = self.getSizeBytes()
        info['seed'] = random.getrandbits(64) # not a real seed but shows that runs are random
        
        return prefixDict(info, prefix=prefix)
    
    
class REQSketch(KLLSketch):
    name = 'REQ'
    
    def __init__(self, size=200, seed=None):
#         if seed is not None:
#             print("REQ sketch does not support seeding rng")

        self.sketch = req_floats_sketch(size)

class TDigestSketch(KLLSketch):
    name = 'TDigest'
    
    def __init__(self, delta=0.01, seed=None):
        self.sketch = TDigest(delta=delta)
    
    def add(self, x):
        self.sketch.update(x)
        
    def query(self, q=None, parameters=None):
        if q == 'quantile':
            return self.sketch.percentile(parameters*100)
        else:
            return self.sketch.cdf(parameters)
    
    def getSizeBytes(self):
        return self.getSize()*16
    
    def getSize(self):
        centroids = self.sketch.centroids_to_list()
        return len(centroids)
    
    
#####################################################
# 
# distinct counting sketches
#

from datasketches import hll_sketch, cpc_sketch, update_theta_sketch
import mmh3

class ThetaSketch(Sketch):
    name = 'Theta'
    
    def __init__(self, lg_k, p, seed=0):
        self.sketch = update_theta_sketch(lg_k, p, seed)
        self.seed = seed
        self.lg_k = lg_k
        
    def salt(self, x):
        if self.seed == 0:
            return x
        elif isinstance(x, int) or isinstance(x, float):
            return mmh3.hash64(bytes(x))[0]
        else:
            return mmh3.hash64(x)[0]
        
    def add(self, x):        
        self.sketch.update(self.salt(x))
        
    def query(self, q=None, parameters=None):
        return self.sketch.get_estimate()
    
    def getSizeBytes(self):
        return self.sketch.get_num_retained()*32
    
    def getSize(self):
        return self.sketch.get_num_retained()

    def info(self, prefix="sketch_"):
        info = super().info(prefix=None)
        info['bytes'] = self.getSizeBytes()
        info['seed'] = self.seed
        
        return prefixDict(info, prefix=prefix)

class HLLSketch(ThetaSketch):
    name = 'HLL'
    
    def __init__(self, lg_k, seed=0):
        self.sketch = hll_sketch(lg_k)
        self.lg_k = lg_k
        self.seed = seed
    
    def getSizeBytes(self):
        return self.sketch.get_updatable_serialization_bytes()
    
    def getSize(self):
        return int(2**self.lg_k)
    
