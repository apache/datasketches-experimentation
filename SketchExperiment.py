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


# Experiments consist of a
# - workload
#   - data generator
#   - query generator
# - oracle
# - sketches
#
# Running an experiment will generate a csv containing the experimental results
# This can be fed into a function/script/notebook to process and plot the results
#
# For an existing problem and new, experimental sketch, one just needs to create a sketch that wraps the 
# implementation with the simple API used in the test. 
#
# For a novel problem, one must also write a query generator and an oracle that will compute the correct answer and calculate the error
# when comparing to a sketch's answer.
#

#
# Replicates in an experiment come from
# - reordering / generating new data sequences (runs)
# - randomization at the sketch level (repetitions)

# To make things parallelizable we
# - For each run, write a workload's data sequence to disk (unless it's just drawing from a distribution)
# - Compute the oracle's answers for that sequence and cache it on disk
# - Create a bunch of jobs to evaluate a sketch on the data sequence using the cached answers
#
# Since our jobs contain nested class instances, python's default pickle doesn't work
# So we use dill to pickle the instances ourselves and unpickle when its run





#############################################################################################################################

import os, psutil
import logging

import time
from random import uniform
import pandas as pd
import numpy as np
from enum import Enum
import copy

import dill # needed to pickle classes/functions for use in multiprocessing 

from collections import deque
import importlib

from concurrent.futures import ProcessPoolExecutor
from experiment_utils import makeDict
        
############   

class ResultAccumulator:
    def __init__(self, save_answers=False):
        self.results = []
        self.timings = []
        self.save_answers = save_answers
        
    def addResult(self, sketch, error, answer, truth, query, workload):
        if not self.save_answers:
            answer = None
        query_info = query.info()
        sketch_info = sketch.info()
        workload_info = workload.info()
        result = makeDict(error=error, answer=answer, truth=truth,
                          **query_info, **workload_info, **sketch_info)
                          #qid=query.qid, **workload_info, **sketch_info)
        self.results.append(result)
    
    def extend(self, results: list):
        self.results.extend(results)
        
    def merge(self, results):
        self.results.extend(results.results)
        self.timings.extend(results.timings)
        
    def toDataFrame(self):
        df = pd.DataFrame(self.results)
        return df    
    
    def addTiming(self, run_time, sketch, workload):
        sketch_info = sketch.info()
        workload_info = workload.info()
        time_result = makeDict(time=run_time, **workload_info, **sketch_info)
        self.timings.append(time_result)
        
    def timingsToDataFrame(self):
        df = pd.DataFrame(self.timings)
        return df
        
class ExperimentOptions:
    class OracleMode(Enum):
        PRECOMPUTE = 0       # standard case which can be serialized to a C++/Java test
        POSTCOMPUTE = 1      # for when the oracle needs a large datastructure to evaluate the error
        JOINTLY_COMPUTE = 2  # likely to lead to bad timings and memory

    def __init__(
        self, 
        ndatasets=1,         # number of times to generate or reorder the dataset
        nrepetitions=10,     # number of times to repeat a given ordering. (so the true answers do not need to be recomputed)
        oracle_mode=OracleMode.PRECOMPUTE,
        nparallel=1,          # number of parallel processes
        maxtasks=1,          # max number of concurrent workload instances
        log_file=None,        
        save_answers=False,
    ):
        self.ndatasets = ndatasets
        self.nrepetitions = nrepetitions
        self.oracle_mode = oracle_mode
        self.nparallel = nparallel
        self.maxtasks = maxtasks
        self.log_file = log_file
        self.save_answers = save_answers
        
    
####################################################################
#    
# Classes used to run parallel processes
# 
# Task = Each data stream in the experiment + the oracle + all the sketches
# Job = every pass through the stream. Each job has its sketch randomized with a different seed
#
# The basic architecture is that an experiment generates Tasks.
# Each Task generates a collection of Jobs
# If the data streams are processed in parallel
#     Each Job uses dill to pickle the Job
#     Each child process unpickles a Job and runs it
#   
# Note that some work may be required to ensure all relevant variables are pickled or to ensure
# the enclosing environment for the unpickled job matches the original environment
#
####################################################################

class Job:
    """
    given an instantiated workload, evaluates the specified sketch
    """
    # assumes workload and oracle are already appropriately initialized    
    def __init__(self, repetition, sketch_name, sketch_fun, sketch_kwargs, workload, oracle, options):
        self.repetition = repetition
        self.sketch_name = sketch_name        
        self.sketch_fun = sketch_fun
        self.sketch_kwargs = sketch_kwargs
        self.sketch_name = sketch_name
        self.workload = workload
        self.oracle = oracle
        self.results = ResultAccumulator(save_answers=True) # XXX 
        self.options = options
                
def executeJob(job):
    job.workload.prepare()
    job.oracle.prepare()
    
    print(f"Running {job.workload.getID()}, repetition {job.repetition}, on sketch {job.sketch_name} {str(job.sketch_kwargs)} ")
    
    query_iter = job.workload.genQueries() # XXX just workload
    q = next(query_iter)
    sketch = job.sketch_fun(seed=job.repetition, **job.sketch_kwargs)
    sketch.name = job.sketch_name
    results = ResultAccumulator(save_answers=job.options.save_answers)
    
    start_time = time.perf_counter()
    for i, x in enumerate(job.workload.genData()):
        sketch.add(x)
                                
        while q and i == q.data_idx:
            answer = sketch.query(q.query, q.parameters)
            error = job.oracle.eval_sketch_answer(q.qid, answer)
            truth = job.oracle.getAnswer(q.qid)
            results.addResult(
                    error=error, 
                    answer=answer, 
                    truth=truth,
                    query=q,
                    sketch=sketch,                     
                    workload=job.workload)
            q = next(query_iter, None)
    run_time = time.perf_counter() - start_time
    results.addTiming(run_time=run_time, sketch=sketch, workload=job.workload)
    return results
        
def executePickledJob(pickled_job):
    job = dill.loads(pickled_job)
    return executeJob(job)

class Task:
    """
    controls multiple runs of a workload
    """
    def __init__(self, run, sketch_seed_start, experiment, executor=None):
        self.run = run
        self.workload = copy.deepcopy(experiment.workload)
        self.oracle = copy.deepcopy(experiment.oracle)
        self.experiment = experiment
        self.executor = executor
        self.sketch_seed_start = sketch_seed_start
        
    # create jobs for each process
    # Currently does not parallelize oracle preparation
    def makeJobs(self):
        print(f"Making jobs for run {self.run}")
        self.workload.reset(seed=self.run)
        self.oracle.reset(workload=self.workload)            
        self.oracle.prepare()
        workload = self.workload
        oracle = self.oracle
            
        workload.prepareForPickle()
        oracle.prepareForPickle()
        for rep in range(self.experiment.options.nrepetitions):
            for sketch_name, (sketch_fun, sketch_params) in self.experiment.sketches.items():
                for p in sketch_params:
                    yield Job(self.sketch_seed_start + rep, sketch_name, sketch_fun, p, workload, oracle, options=self.experiment.options)
                
                
####################################################################################################                

class SketchExperiment:
    """
    Run an experiment for given workload over a collection of sketches
    The workloads can be randomized with a seed or they can be repeated for randomized sketches
    """
    
    name = "BaseExperiment"
    target_results = None # iterable providing the corresponding results (or a way to compute the result)
    sketches = None # dictionary of sketches
    MAX_SKETCHES = 100000
    
    def __init__(self, workload, oracle, options=None, result_file="exp_results.csv"):
        # basic setup variables
        self.workload = workload
        self.oracle = oracle        
        self.options = options if options is not None else ExperimentOptions()
        self.results = ResultAccumulator()
        self.timer = {}
        self.result_file=result_file
        self.start_seed = 0
    
    def getNumSketches(self):
        num = 1
        for v in self.experiment.sketches.values():
            num *= len(v)
        return num
        
    def addSketches(self, sketches):
        self.sketches = sketches
        
    def setOptions(self, options):
        self.options = options
                
    def prepareOracle(self, size=None, **kwargs):
        raise Exception

    def prepare(self, **kwargs):
        logging.info("prep workload0")
        self.workload.prepare()         
    
    def execute(self, write_mode="w"):        
        assert(self.options.oracle_mode == self.options.OracleMode.PRECOMPUTE) 
    
        if self.options.nparallel > 1:
            executor = ProcessPoolExecutor(max_workers=self.options.nparallel)

        for run in range(self.options.ndatasets):
            logging.info(f"Starting run {run}")
            workload_seed = self.start_seed + run
            sketch_seed_start = workload_seed * self.MAX_SKETCHES
            task = Task(workload_seed, sketch_seed_start, experiment=self) 
            futures = deque()
            job_sizes = []
            for job in task.makeJobs():
                if self.options.nparallel > 1:
                    pickled_job = dill.dumps(job)
                    job_sizes.append(len(pickled_job))
                    futures.append(executor.submit(executePickledJob, pickled_job))
                else:                    
                    job_result = executeJob(job)
                    self.results.merge(job_result)
            job_sizes = np.array(job_sizes)
            print(f"pickled job max size: {max(job_sizes)} avg size: {np.mean(job_sizes)}")
            if self.options.nparallel > 1:
                # block when there are enough parallel jobs running
                while futures and len(futures) >= self.options.nparallel:                    
                    f = futures.popleft()
                    job_result = f.result()
                    self.results.merge(job_result)

        # cleanup last jobs
        while futures:
            f = futures.popleft()
            job_result = f.result()
            self.results.merge(job_result)

        print(f"writing output {self.result_file} mode: {write_mode}")
        self.results.toDataFrame().to_csv(self.result_file, mode=write_mode, header=(write_mode == 'w'))
            
        timing_df = self.results.timingsToDataFrame()
        agg_timing_df = timing_df.groupby(['sketch_name']).agg(runtime=('time', np.mean))
        print(agg_timing_df.to_string())
    # TODO: clean up the temporary answer and pickle files
  
####################################################################################################                

class SketchMetaExperiment:
    """
    Run experiments on multiple workloads
    """
    def __init__(self, workloads, oracle, options=None, result_file="exp_results.csv"):
        self.workloads = workloads
        self.oracle = oracle
        self.options = options
        self.result_file=result_file
        self.sketches = []
        
    def execute(self, default_write_mode='w'):
        write_mode = default_write_mode
        for w in self.workloads:
            self.oracle.reset(workload=w)
            e = SketchExperiment(workload=w, 
                                 oracle=self.oracle,
                                 options=self.options,
                                 result_file=self.result_file)
            e.addSketches(self.sketches)
            e.prepare()
            e.execute(write_mode=write_mode)
            write_mode = "a"
            
    def addSketches(self, sketches):
        self.sketches = sketches
        