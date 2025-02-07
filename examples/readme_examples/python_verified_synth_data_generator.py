# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from synth_gen.generators import PythonVerifiedSynthDataGenerator

generator = PythonVerifiedSynthDataGenerator()
generator.verbose = True
generations = generator.generate_training_data(seed="Fibonacci with a twist.")
print(generations[0])
