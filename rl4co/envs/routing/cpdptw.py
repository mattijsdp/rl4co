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
    """(Capacitated) Pickup and Delivery Problem with Time Windows (PDPTW) environment.
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

    name = "cpdptw"

    def __init__(
        self,
        max_loc: float = 150,
        max_time: int = 480,
        scale: bool = True,
        **kwargs,
    ):
        self.max_loc = max_loc
        self.min_time = 0
        self.max_time = max_time
        self.scale = scale
        super().__init__(**kwargs)

    def _make_spec(self, td_params: Optional[TensorDict] = None):
        super()._make_spec(td_params)

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
        if batch_size is None:
            batch_size = self.batch_size if td is None else td.batch_size
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)

        self.to(td.device)

        locs = torch.cat((td["depot"][:, None, :], td["locs"]), -2)

        # Pick is 1, deliver is 0 [batch_size, graph_size+1], [1,1...1, 0...0]
        to_deliver = torch.cat(
            [
                torch.ones(
                    *batch_size,
                    self.num_loc // 2 + 1,
                    dtype=torch.bool,
                    device=self.device,
                ),
                torch.zeros(
                    *batch_size, self.num_loc // 2, dtype=torch.bool, device=self.device
                ),
            ],
            dim=-1,
        )

        # Cannot visit depot at first step # [0,1...1] so set not available
        available = torch.ones(
            (*batch_size, self.num_loc + 1), dtype=torch.bool, device=self.device
        )
        action_mask = ~available.contiguous()  # [batch_size, graph_size+1]
        action_mask[..., 0] = 1  # First step is always the depot

        # Other variables
        current_node = torch.zeros(
            (*batch_size, 1), dtype=torch.int64, device=self.device
        )
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=self.device)

        return TensorDict(
            {
                "locs": locs,
                "current_node": current_node,
                "to_deliver": to_deliver,
                "available": available,
                "i": i,
                "action_mask": action_mask,
                "current_time": torch.zeros(
                    *batch_size, 1, dtype=torch.float32, device=self.device
                ),
                "durations": td["durations"],
                "time_windows": td["time_windows"],
                "distances": td["distances"],
            },
            batch_size=batch_size,
        )
    
    # @staticmethod
    def get_action_mask(self, td: TensorDict) -> TensorDict:
        action_mask = super(CPDPTWEnv, CPDPTWEnv).get_action_mask(td)

        batch_size = td["locs"].shape[0]
        current_loc = gather_by_index(td["locs"], td["current_node"]).reshape(
            [batch_size, 2]
        )
        dist = get_distance(current_loc, td["locs"].transpose(0, 1)).transpose(0, 1)
        td.update({"distances": dist})
        can_reach_in_time = (
            td["current_time"] + dist <= td["time_windows"][..., 1]
        )  # I only need to start the service before the time window ends, not finish it.
        return action_mask & can_reach_in_time
    
    def _step(self, td: TensorDict) -> TensorDict:
        """In addition to the calculations in the PDPEnv, the current time is
        updated to keep track of which nodes are still reachable in time.
        The current_node is updated in the parent class' _step() function.
        """
        batch_size = td["locs"].shape[0]
        # update current_time
        distance = gather_by_index(td["distances"], td["action"]).reshape([batch_size, 1])
        duration = gather_by_index(td["durations"], td["action"]).reshape([batch_size, 1])
        start_times = gather_by_index(td["time_windows"], td["action"])[..., 0].reshape(
            [batch_size, 1]
        )
        td["current_time"] = (td["action"][:, None] != 0) * (
            torch.max(td["current_time"] + distance, start_times) + duration
        )
        # current_node is updated to the selected action
        td = super()._step(td)
        return td
    
    def get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        print("reward")
        return torch.randn(td.batch_size, device=td.device, dtype=torch.float32)
        # return super().get_reward(td, actions)
    
    # @staticmethod
    def check_solution_validity(self, td: TensorDict, actions: torch.Tensor):
        super().check_solution_validity(td, actions)
        batch_size = td["locs"].shape[0]
        # distances to depot
        distances = get_distance(
            td["locs"][..., 0, :], td["locs"].transpose(0, 1)
        ).transpose(0, 1)
        # basic checks on time windows
        assert torch.all(distances >= 0.0), "Distances must be non-negative."
        assert torch.all(td["time_windows"] >= 0.0), "Time windows must be non-negative."
        assert torch.all(
            td["time_windows"][..., :, 0] + distances + td["durations"]
            <= td["time_windows"][..., 0, 1][0]  # max_time is the same for all batches
        ), "vehicle cannot perform service and get back to depot in time."
        assert torch.all(
            td["durations"] >= 0.0
        ), "Service durations must be non-negative."
        assert torch.all(
            td["time_windows"][..., 0] < td["time_windows"][..., 1]
        ), "there are unfeasible time windows"
        # check vehicles can meet deadlines
        curr_time = torch.zeros(batch_size, 1, dtype=torch.float32, device=td.device)
        curr_node = torch.zeros_like(curr_time, dtype=torch.int64, device=td.device)
        for ii in range(actions.size(1)):
            next_node = actions[:, ii]
            dist = get_distance(
                gather_by_index(td["locs"], curr_node).reshape([batch_size, 2]),
                gather_by_index(td["locs"], next_node).reshape([batch_size, 2]),
            ).reshape([batch_size, 1])
            curr_time = torch.max(
                (curr_time + dist).int(),
                gather_by_index(td["time_windows"], next_node)[..., 0].reshape(
                    [batch_size, 1]
                ),
            )
            assert torch.all(
                curr_time
                <= gather_by_index(td["time_windows"], next_node)[..., 1].reshape(
                    [batch_size, 1]
                )
            ), "vehicle cannot start service before deadline"
            curr_time = curr_time + gather_by_index(td["durations"], next_node).reshape(
                [batch_size, 1]
            )
            curr_node = next_node
            curr_time[curr_node == 0] = 0.0  # reset time for depot

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
                "distances": dist,
            }
        )
        return td

    @staticmethod
    def render(td: TensorDict, actions=None, ax=None):
        return PDPEnv.render(td=td, actions=actions, ax=ax)