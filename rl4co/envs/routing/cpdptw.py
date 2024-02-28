from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.routing.pdp import PDPEnv
from rl4co.utils.ops import gather_by_index, get_tour_length, get_distance


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
    and must visit both within their respective time windows.

    Args:
        num_loc: number of locations (cities)
        min_loc: minimum value for the locations both x and y
        max_loc: maximum value for the locations both x and y
        td_params: parameters of the environment
        seed: seed for the environment
        device: device to use.  Generally, no need to set as tensors are updated on the fly
    """

    name = "pdptw"

    def __init__(
        self,
        max_loc: float = 150,
        max_time: int = 480,
        scale: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_loc = max_loc
        self.min_time = 0
        self.max_time = max_time
        self.scale = scale

    def generate_data(self, batch_size) -> TensorDict:
        td = super().generate_data(batch_size)

        batch_size = td.batch_size

        ## define service durations
        # generate randomly (first assume service durations of 0, to be changed later)
        durations = torch.zeros(
            *batch_size, self.num_loc + 1, dtype=torch.float32, device=self.device
        )

        ## define time windows
        # 1. get distances from depot
        dist = get_distance(td["depot"], td["locs"].transpose(0, 1)).transpose(0, 1)
        dist = torch.cat((torch.zeros(*batch_size, 1, device=self.device), dist), dim=1)
        # 2. define upper bound for time windows to make sure the vehicle can get back to the depot in time
        # TODO: check upper_bound is not negative?
        upper_bound = self.max_time - dist - durations
        # 3. create random values between 0 and 1
        ts_1 = torch.rand(*batch_size, self.num_loc + 1, device=self.device)
        ts_2 = torch.rand(*batch_size, self.num_loc + 1, device=self.device)
        # 4. scale values to lie between their respective min_time and max_time and convert to integer values
        min_ts = (dist + (upper_bound - dist) * ts_1).int()
        max_ts = (dist + (upper_bound - dist) * ts_2).int()
        # 5. set the lower value to min, the higher to max
        min_times = torch.min(min_ts, max_ts)
        max_times = torch.max(min_ts, max_ts)
        # 6. reset times for depot
        min_times[..., :, 0] = 0.0
        max_times[..., :, 0] = self.max_time

        # 7. ensure min_times < max_times to prevent numerical errors in attention.py
        # min_times == max_times may lead to nan values in _inner_mha()
        mask = min_times == max_times
        if torch.any(mask):
            min_tmp = min_times.clone()
            min_tmp[mask] = torch.max(
                dist[mask].int(), min_tmp[mask] - 1
            )  # we are handling integer values, so we can simply substract 1
            min_times = min_tmp

            mask = min_times == max_times  # update mask to new min_times
            if torch.any(mask):
                max_tmp = max_times.clone()
                max_tmp[mask] = torch.min(
                    torch.floor(upper_bound[mask]).int(),
                    torch.max(
                        torch.ceil(min_tmp[mask] + durations[mask]).int(),
                        max_tmp[mask] + 1,
                    ),
                )
                max_times = max_tmp

        # scale to [0, 1]
        if self.scale:
            durations = durations / self.max_time
            min_times = min_times / self.max_time
            max_times = max_times / self.max_time
            # TODO: does this assume 1:1 distance to time ratio?
            td["depot"] = td["depot"] / self.max_time
            td["locs"] = td["locs"] / self.max_time

        # 8. stack to tensor time_windows
        time_windows = torch.stack((min_times, max_times), dim=-1)

        assert torch.all(
            min_times < max_times
        ), "Please make sure the relation between max_loc and max_time allows for feasible solutions."

        # reset duration at depot to 0
        durations[:, 0] = 0.0
        td.update(
            {
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
