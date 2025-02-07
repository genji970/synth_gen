# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from synth_gen.generation.seed_source import SmallPythonSeedSource


def test_seed():
    source = SmallPythonSeedSource()
    seed = source.get_next_seed()
    print("seed:")
    print(seed)
    assert seed is not None
    assert len(seed) > 0
    assert type(seed) is str
    assert len(seed) < 1000


if __name__ == "__main__":
    test_seed()
