#%load_ext autoreload

import Sampler
import DataGenerator
import SketchExperiment
import logging
import Oracle

logging.getLogger().setLevel(logging.WARNING)

import argparse

def runInstacartExperiment(prarser):
    w = DataGenerator.InstacartTopKWorkload(k=10, num_queries=30)
    oracle = Oracle.TopKOracle(workload=w)
    e = SketchExperiment.SketchExperiment(workload=w, 
                                          oracle=oracle,
                                          sketches=
                                          repetitions=3)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TopK experiments")
    parser.add_argument('--nrep', default=1)
    parser.add_argument('--seed', default=0)
    parser.add_argument('-k', default=10)
    parser.add_argument('--maxsize', default=30)
    parser.add_argument('--numqueries', default=10)
    parser.add_argument('--experiment', default='PitmanYor')

    parser.parse_args()

    sketches = {'TopKSampler':lambda: TopKSamplerSketch(k=10, maxsize=20)},
    
    if experiment == 'Instacart':
        
    elif experiment == 'PitmanYor':
        w = DataGenerator.PitmanYorWorkload(length=int(1e5), alpha=1,beta=0.5, k=10, num_queries=10)
    elif experiment == 'StackOverflow':
        pass
    elif experiment == 'retail':
        pass

