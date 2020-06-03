#!/bin/bash

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

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $SCRIPT_DIR

function prepare {
    # Remove old directories and clone repo
    cd $SCRIPT_DIR
    rm -rf shakespeare_data leaf

    # Clone leaf repo
    git clone https://github.com/TalwalkarLab/leaf
}

function generate {
    # Generate dataset
    cd leaf/data/shakespeare
    ./preprocess.sh -s niid --sf 1.0 -k 0 -t sample -tf 0.8 --smplseed 1591177658

    # Go back to this scripts root directory
    cd $SCRIPT_DIR
    mv leaf/data/shakespeare/data shakespeare_data

}

function cleanup {
    # Cleanup repo
    cd $SCRIPT_DIR
    rm -rf leaf
}

prepare
generate
cleanup
