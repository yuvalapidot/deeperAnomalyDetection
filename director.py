import logging

import memory_slice_generator as generation

logging.basicConfig(filename='DeeperAnomalyDetector.log', level=logging.DEBUG,
                    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')

logger = logging.getLogger("Director")

generator = generation.MemorySliceGenerator(["D:\DeepFeaturesExperiment\Dumps"], 1048576)

for X, y in generator.generate_memory_slices():
    pass
