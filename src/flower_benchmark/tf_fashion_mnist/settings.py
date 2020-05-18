# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides a variaty of benchmark settings."""

from dataclasses import dataclass


@dataclass
class Setting:
    """One specific training setting."""

    # Global paramters
    rounds: int = 100

    # Server paramters
    sample_fraction: float = 1.0
    min_sample_size: int = 100
    min_num_clients: int = 100
    training_round_timeout: int = 3600

    # Client parameters
    iid_fraction: float
    num_partitions: int = 100
    delay_factor: float


def get_setting(name: str) -> Setting:
    """Return appropriate setting."""
    if name not in SETTINGS:
        raise Exception(
            "Setting does not exist. Valid settings are: %s" % list(SETTINGS.keys())
        )

    return SETTINGS[name]


SETTINGS = {
    "minimal": Setting(
        rounds=2,
        sample_fraction=1.0,
        min_sample_size=1,
        min_num_clients=1,
        training_round_timeout=3600,
        iid_fraction=0.0,
        num_partitions=1,
        delay_factor=0.0,
    )
}

# Parameterized iid_fraction
for i in range(11):
    # Parameterized timeout
    for timeout in [20, 60]:
        SETTINGS[f"delay_0.0_timeout_{timeout}_fraction_{i/10}"] = Setting(
            iid_fraction=i / 10, training_round_timeout=timeout
        )
