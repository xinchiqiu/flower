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
"""Provides functions to construct various flower CLI commands."""

from typing import Optional


def install_wheel(wheel_remote_path: str) -> str:
    """Return install command for wheel.

    Remove previous versions if existing.
    """
    return (
        "python3.7 -m pip uninstall -y flower && "
        + f"python3.7 -m pip install '{wheel_remote_path}[examples-tensorflow,http-logger]'"
    )


def start_logserver(
    logserver_s3_bucket: Optional[str] = None, logserver_s3_key: Optional[str] = None
) -> str:
    """Return command to run logserver."""
    cmd = "screen -d -m python3.7 -m flower_logserver"

    if logserver_s3_bucket is not None and logserver_s3_key is not None:
        cmd += f" --s3_bucket={logserver_s3_bucket}" + f" --s3_key={logserver_s3_key}"

    return cmd


# pylint: disable=too-many-arguments
def start_server(log_host: str, benchmark: str, setting: str) -> str:
    """Build command to run server."""
    return (
        "screen -d -m"
        + f" python3.7 -m flower_benchmark.{benchmark}.server"
        + f" --log_host={log_host}"
        + f" --setting={setting}"
    )


def start_client(
    grpc_server_address: str, log_host: str, benchmark: str, setting: str, index: int
) -> str:
    """Build command to run client."""
    return (
        "screen -d -m"
        + f" python3.7 -m flower_benchmark.{benchmark}.client"
        + f" --grpc_server_address={grpc_server_address}"
        + f" --log_host={log_host}"
        + f" --setting={setting}"
        + f" --index={index}"
    )


def download_dataset(benchmark: str) -> str:
    "Return command which makes dataset locally available."
    return f"python3.7 -m flower_benchmark.{benchmark}.download"


def watch_and_shutdown(keyword: str, adapter: str) -> str:
    """Return command which shuts down the instance after no benchmark is running anymore."""
    cmd = f"screen -d -m bash -c 'while [[ $(ps a | grep {keyword}) ]]; do sleep 1; done; "

    if adapter == "docker":
        cmd += "sleep 300 && kill 1'"
    elif adapter == "ec2":
        # Shutdown after 2 minutes to allow a logged in user
        # to chancel the shutdown manually just in case
        cmd += "sudo shutdown -P 2'"
    else:
        raise Exception("Unknown Adapter")

    return cmd
