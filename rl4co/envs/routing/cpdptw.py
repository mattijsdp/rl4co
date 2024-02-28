from typing import Optional
import torch
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)


class CPDPTWEnv(PDPEnv):
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

        durations = BoundedTensorSpec(
            low=self.min_time,
            high=self.max_time,
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
                **self.observation_spec,
                "current_time": current_time,
                "durations": durations,
                "time_windows": time_windows,
                # vehicle_idx=vehicle_idx,
            }
        )

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[int] = None
    ) -> TensorDict:
        """Reset the environment to an initial state.

        Args:
            td: tensor dictionary containing the parameters of the environment
            batch_size: batch size for the environment

        Returns:
            Tensor dictionary containing the initial observation
        """

        td_reset = super()._reset(td, batch_size)
        td_reset.update(
            {
                "current_time": torch.zeros(
                    *batch_size, 1, dtype=torch.float32, device=self.device
                ),
                "durations": td["durations"],
                "time_windows": td["time_windows"],
            }
        )
        return td_reset
