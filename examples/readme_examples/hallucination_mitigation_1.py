# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from synth_gen.generators import UnkownAnswerGenerator

wikipedia_article = """
Albert Einstein was a German-born theoretical physicist who is best known for developing the theory of relativity."""

generator = UnkownAnswerGenerator()

generator.verbose = True
generations = generator.generate_training_data(wikipedia_article)
