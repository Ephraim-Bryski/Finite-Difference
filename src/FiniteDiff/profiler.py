
from FD import *
import cProfile
import pstats
import os


from FD import *
import numpy as np

def run_model():
    # all my code to test
    pass


f_name = "boop.prof"


with cProfile.Profile() as profile:
    run_model()

results = pstats.Stats(profile)

results.dump_stats(f_name)
os.system(f"snakeviz {f_name}")




