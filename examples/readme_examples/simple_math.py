# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from synth_gen.generators import SimpleMathTimesGenerator

generator = SimpleMathTimesGenerator()

for t in range(100):
    generations = generator.generate_training_data()
    print("Human: ", generations[0].chat.messages[0].content)
    print("AI: ", generations[0].chat.messages[1].content)
