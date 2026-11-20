from synth_gen.generation.seed_source import SmallPythonSeedSource
from synth_gen.generators import PythonVerifiedSynthDataGenerator

generator = PythonVerifiedSynthDataGenerator()
generator.seed_source = SmallPythonSeedSource()
generator.verbose = True
generation = generator.generate_training_data(seed="Fibonacci with a twist.")
print(generation[0])