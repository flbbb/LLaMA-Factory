# Copyright 2024 the LlamaFactory team.
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

import idr_torch
from dotenv import load_dotenv
from torch import distributed as dist

from llamafactory.train.tuner import run_exp


def launch():
    load_dotenv()
    # if dist.is_initialized():
    #     print("Distributed environment already initialized.")
    # else:
    #     print("Initializing distributed environment...")
    #     dist.init_process_group(
    #         backend="nccl",
    #         init_method="env://",
    #         world_size=idr_torch.size,
    #         rank=idr_torch.rank,
    #     )
    run_exp()


if __name__ == "__main__":
    launch()
