from math import sqrt
from typing import Optional
import torch
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)

from rl4co.envs.routing.cvrp import PDPEnv, CAPACITIES
from rl4co.utils.ops import gather_by_index, get_distance
from rl4co.data.utils import (
    load_npz_to_tensordict,
    load_solomon_instance,
    load_solomon_solution,
)


Class CPDPTWEnv(PDPEnv):
    """(Capacitated) Pickup and Delivery Problem with Time Windows (CPDPTW) environment.
    Inherits from the PDPEnv class in which capacities and time windows are added.
    Additionally considers time windows within which a pickups have to be completed.
    The environment is made of num_loc + 1 locations (cities):
        - 1 depot
        - `num_loc` / 2 pickup locations
        - `num_loc` / 2 delivery locations
    The goal is to visit all the pickup and delivery locations in the shortest path possible starting from the depot
    The conditions is that the agent must visit a pickup location before visiting its corresponding delivery location

    Args:
        num_loc: number of locations (cities) in the TSP
        td_params: parameters of the environment
        seed: seed for the environment
        device: device to use.  Generally, no need to set as tensors are updated on the fly
        vehicle_capacity: capacity of the vehicle
        max_time: maximum time to complete the tour
    """

    name = "cpdptw"

    def __init__(
        self,
        max_loc: float = 150,  # different default value to PDPEnv to match max_time, will be scaled
        max_time: int = 480,
        **kwargs,
    ):
        self.min_time = 0  # always 0
        self.max_loc = max_loc
        self.max_time = max_time
        super().__init__(max_loc=max_loc, max_time=max_time, **kwargs)
    
    def _make_spec(self, td_params: Optional[TensorDict] = None):
        super().make_spec(td_params)

        current_time = UnboundedContinuousTensorSpec(
            shape=(1), dtype=torch.float32, device=self.device
        )

        current_loc = UnboundedContinuousTensorSpec(
            shape=(2), dtype=torch.float32, device=self.device
        )

        durations = BoundedTensorSpec(
            low=self.min_time,
            high=self.min_time,
            shape=(self.num_loc, 1),
            dtype=torch.int64,
            device=self.device,
        )

        time_windows = BoundedTensorSpec(
            low=self.min_time,
            high=self.max_time,
            shape=(
                self.num_loc, 2
            ),  # each location has a 2D time window (start, end)
            dtype=torch.int64,
            device=self.device,
        )

        # extend observation specs
        self.observation_spec = CompositeSpec(
            {
                "current_time": current_time,
                "current_loc": current_loc,
                "durations": durations,
                "time_windows": time_windows,
                # vehicle_idx=vehicle_idx,
                **self.observation_spec,
            }
        )


    def _reset(
        
    )
        