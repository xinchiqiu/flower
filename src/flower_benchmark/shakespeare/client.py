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
import argparse
from logging import ERROR
from typing import Tuple, Dict

import numpy as np
import tensorflow as tf

import flower as fl
from flower.logger import configure, log
from flower_benchmark.common import load_partition

# from flower_benchmark.dataset import tf_cifar_partitioned
from flower_benchmark.model import stacked_lstm
from flower_benchmark.tf_shakespeare.load_data import load_data
from flower_benchmark.tf_shakespeare.settings import SETTINGS, get_setting
from flower_benchmark.common import keras_evaluate
from . import DEFAULT_GRPC_SERVER_ADDRESS, DEFAULT_GRPC_SERVER_PORT, SEED

tf.get_logger().setLevel("ERROR")


class ShakespeareClient(fl.Client):
    def __init__(
        self,
        cid: str,
        model: tf.keras.Model,
        xy_train: Tuple[np.ndarray, np.ndarray],
        xy_test: Tuple[np.ndarray, np.ndarray],
        delay_factor: float,
        num_classes: int,
    ):

        super().__init__(cid)
        self.model = model

        self.ds_train = build_dataset(
            xy_train[0],
            xy_train[1],
            num_classes=num_classes,
            shuffle_buffer_size=len(xy_train[0]),
            augment=False,
        )
        self.ds_test = build_dataset(
            xy_test[0],
            xy_test[1],
            num_classes=num_classes,
            shuffle_buffer_size=0,
            augment=False,
        )

        self.num_examples_train = len(xy_train[0])
        self.num_examples_test = len(xy_test[0])
        self.delay_factor = delay_factor

    def get_parameters(self) -> fl.ParametersRes:
        parameters = fl.weights_to_parameters(self.model.get_weights())
        return fl.ParametersRes(parameters=parameters)
    def fit(self, ins: flwr.FitIns) -> flwr.FitRes:
        weights: fl.Weights = fl.parameters_to_weights(ins[0])
        config = ins[1]
        log(
            DEBUG,
            "fit on %s (examples: %s), config %s",
            self.cid,
            self.num_examples_train,
            config,
        )

        # Training configuration
        # epoch_global = int(config["epoch_global"])
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        # lr_initial = float(config["lr_initial"])
        # lr_decay = float(config["lr_decay"])
        timeout = int(config["timeout"]) if "timeout" in config else None
        partial_updates = bool(int(config["partial_updates"]))

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Train the local model using the local dataset
        completed, fit_duration, num_examples = custom_fit(
            model=self.model,
            dataset=self.ds_train,
            num_epochs=epochs,
            batch_size=batch_size,
            callbacks=[],
            delay_factor=self.delay_factor,
            timeout=timeout,
        )
        log(DEBUG, "client %s had fit_duration %s", self.cid, fit_duration)

        # Compute the maximum number of examples which could have been processed
        num_examples_ceil = self.num_examples_train * epochs

        if not completed and not partial_updates:
            # Return empty update if local update could not be completed in time
            parameters = fl.weights_to_parameters([])
        else:
            # Return the refined weights and the number of examples used for training
            parameters = fl.weights_to_parameters(self.model.get_weights())
        return parameters, num_examples, num_examples_ceil, fit_duration

    def evaluate(self, ins: fl.EvaluateIns) -> fl.EvaluateRes:
        weights: fl.Weights = fl.parameters_to_weights(ins[0])
        config: Dict[str, str] = ins[1]
        log(
            DEBUG,
            "evaluate on %s (examples: %s), config %s",
            self.cid,
            self.num_examples_test,
            config,
        )
        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Evaluate the updated model on the local dataset
        loss, acc = keras_evaluate(
            self.model, self.ds_test, batch_size=self.num_examples_test
        )

        # Return the number of evaluation examples and the evaluation result (loss)
        return self.num_examples_test, loss, acc

def parse_args() -> argparse.Namespace:
    """Parse and return commandline arguments."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (IPv6, default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--log_host", type=str, help="HTTP log handler host (no default)",
    )
    parser.add_argument(
        "--setting", type=str, choices=SETTINGS.keys(), help="Setting to run.",
    )
    parser.add_argument("--cid", type=str, required=True, help="Client cid.")
    return parser.parse_args()


def get_client_setting(setting: str, cid: str) -> ClientSetting:
    """Return client setting based on setting name and cid."""
    for client_setting in get_setting(setting).clients:
        if client_setting.cid == cid:
            return client_setting

    raise ClientSettingNotFound()
'''
def parse_args() -> argparse.Namespace:
    """Parse and return commandline arguments."""
    parser = argparse.ArgumentParser(description="Flower Shakespeare")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_GRPC_SERVER_ADDRESS,
        help="gRPC server address (IPv6, default: [::])",
    )
    parser.add_argument(
        "--log_host", type=str, help="HTTP log handler host (no default)",
    )
    parser.add_argument(
        "--setting", type=str, choices=SETTINGS.keys(), help="Setting to run.",
    )
    parser.add_argument(
        "--index", type=int, required=True, help="Client index in settings."
    )
    return parser.parse_args()
'''

def main() -> None:
    """Load data, create and start client."""
    args = parse_args()

    client_setting = get_setting(args.setting).clients[args.index]

    # Configure logger
    configure(identifier=f"client:{client_setting.cid}", host=args.log_host)
    log(INFO, "Starting client, settings: %s", client_setting)

    # Load model
    model = stacked_lstm(
        input_len=80, hidden_size=256, num_classes=80, seed=SEED
    )

    # need to download and preprocess the dataset, make sure to have 2 .json data one for training and one for testing
    # 660 clients, change the client cid from string to int, then in the load_data function can directly find the client and its dataset.
    
    xy_train, xy_test = load_data(
        "../dataset/shakespeare/train",
        "../dataset/shakespeare/test",
        int(client_setting.cid),
    )

    # Start client
    client = ShakespeareClient(
        client_setting.cid, model, xy_train, xy_test, client_setting.delay_factor, 80
    )

    fl.app.start_client(args.server_address, client)


if __name__ == "__main__":
    # pylint: disable=broad-except
    try:
        main()
    except Exception as err:
        log(ERROR, "Fatal error in main")
        log(ERROR, err, exc_info=True, stack_info=True)

        # Raise the error again so the exit code is correct
        raise err
