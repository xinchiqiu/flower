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
"""Federating: Fast and Slow."""


import statistics
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np

from flower.client_manager import ClientManager
from flower.client_proxy import ClientProxy
from flower.typing import EvaluateRes, FitIns, FitRes, Weights

from .aggregate import aggregate, weighted_loss_avg
from .fedavg import FedAvg
from .parameter import parameters_to_weights, weights_to_parameters


class FastAndSlow(FedAvg):
    """Strategy implementation which alternates between fast and slow rounds."""

    # pylint: disable-msg=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 1,
        min_eval_clients: int = 1,
        min_available_clients: int = 1,
        eval_fn: Optional[Callable[[Weights], Optional[Tuple[float, float]]]] = None,
        min_completion_rate_fit: float = 0.5,
        min_completion_rate_evaluate: float = 0.5,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, str]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, str]]] = None,
        r_fast: int = 1,
        r_slow: int = 1,
        t_fast: int = 10,
        t_slow: int = 10,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
        )
        self.min_completion_rate_fit = min_completion_rate_fit
        self.min_completion_rate_evaluate = min_completion_rate_evaluate
        self.r_fast = r_fast
        self.r_slow = r_slow
        self.t_fast = t_fast
        self.t_slow = t_slow
        self.contributions: Dict[str, List[Tuple[int, float]]] = {}

    # pylint: disable-msg=too-many-locals
    def on_configure_fit(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Block until `min_num_clients` are available
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        success = client_manager.wait_for(num_clients=min_num_clients, timeout=60)
        if not success:
            # Do not continue if not enough clients are available
            return []

        # Prepare parameters and config
        parameters = weights_to_parameters(weights)
        config = {}
        if self.on_fit_config_fn is not None:
            # Use custom fit config function if provided
            config = self.on_fit_config_fn(rnd)
        use_fast_timeout = is_fast_round(rnd, self.r_fast, self.r_slow)
        config["timeout"] = str(self.t_fast if use_fast_timeout else self.t_slow)
        fit_ins = (parameters, config)

        # Get all clients and gather their contributions
        all_clients: Dict[str, ClientProxy] = client_manager.all()
        cid_idx: Dict[int, str] = {}
        logits: List[float] = []
        for idx, (cid, _) in enumerate(all_clients.items()):
            cid_idx[idx] = cid
            penalty = 0.0
            if cid in self.contributions.keys():
                contribs: List[Tuple[int, float]] = self.contributions[cid]
                penalty = statistics.mean([c for _, c in contribs])
            # `p` should be:
            #   - High for clients which have never been picked before
            #   - Medium for clients which have contributed, but not used their entire budget
            #   - Low (but not 0) for clients which have been picked and used their budget
            logits.append(1.1 - penalty)

        # Sample clients
        indices = np.arange(len(all_clients.keys()))
        probs = softmax(np.array(logits))
        idxs = np.random.choice(indices, size=sample_size, replace=False, p=probs)
        clients = [all_clients[cid_idx[idx]] for idx in idxs]

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def on_aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None

        # Check if enough results are available
        completion_rate = len(results) / (len(results) + len(failures))
        if completion_rate < self.min_completion_rate_fit:
            # Not enough results for aggregation
            return None

        # Convert results
        weights_results = [
            (parameters_to_weights(parameters), num_examples)
            for client, (parameters, num_examples, _) in results
        ]
        weights_prime = aggregate(weights_results)

        # Track contributions to the global model
        for client, fit_res in results:
            cid = client.cid
            contribution: Tuple[int, float] = (rnd, fit_res[1] / fit_res[2])
            if cid not in self.contributions.keys():
                self.contributions[cid] = []
            self.contributions[cid].append(contribution)

        return weights_prime

    def on_aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Check if enough results are available
        completion_rate = len(results) / (len(results) + len(failures))
        if completion_rate < self.min_completion_rate_evaluate:
            # Not enough results for aggregation
            return None

        return weighted_loss_avg([evaluate_res for _, evaluate_res in results])


def is_fast_round(rnd: int, r_fast: int, r_slow: int) -> bool:
    """Determine if the round is fast or slow."""
    remainder = rnd % (r_fast + r_slow)
    return remainder - r_fast < 0


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax."""
    e_x = np.exp(logits - np.max(logits))
    return cast(np.ndarray, e_x / e_x.sum(axis=0))
