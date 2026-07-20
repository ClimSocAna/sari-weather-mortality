from collections.abc import Sequence

import torch

KREIS_COUNT_BY_LAND = (15, 1, 45, 2, 53, 26, 36, 44, 96, 6, 1, 18, 8, 13, 14, 22)
COARSE_AGE_GROUPS = (tuple(range(8)), (8, 9), (10, 11), (12, 13, 14))
KREIS_COUNT = sum(KREIS_COUNT_BY_LAND)
LAND_COUNT = len(KREIS_COUNT_BY_LAND)


def age_grouping(data: torch.Tensor, axis: int, groups: Sequence[Sequence[int]] = COARSE_AGE_GROUPS) -> torch.Tensor:
    axis_size = data.size(axis)
    for group in groups:
        if not group:
            raise ValueError("Age groups must not be empty")
        if any(index < 0 or index >= axis_size for index in group):
            raise ValueError(f"Age group index out of range for axis size {axis_size}: {group}")

    group_indices = [torch.as_tensor(group, device=data.device, dtype=torch.long) for group in groups]
    grouped_data = [torch.index_select(data, axis, group).sum(dim=axis) for group in group_indices]
    return torch.stack(grouped_data, dim=axis)


def land_grouping(data: torch.Tensor, land_axis: int = 1) -> torch.Tensor:
    if data.size(land_axis) != KREIS_COUNT:
        raise ValueError(f"Expected {KREIS_COUNT} districts on axis {land_axis}, got {data.size(land_axis)}")
    split_tensors = torch.split(data, KREIS_COUNT_BY_LAND, dim=land_axis)
    grouped_tensors = [group.sum(dim=land_axis) for group in split_tensors]
    return torch.stack(grouped_tensors, dim=land_axis)


def land_extension(grouped_x: torch.Tensor, land_axis: int = 1) -> torch.Tensor:
    if grouped_x.size(land_axis) != LAND_COUNT:
        raise ValueError(f"Expected {LAND_COUNT} states on axis {land_axis}, got {grouped_x.size(land_axis)}")
    kreis_count_by_land = torch.as_tensor(KREIS_COUNT_BY_LAND, device=grouped_x.device)
    return torch.repeat_interleave(grouped_x, kreis_count_by_land, dim=land_axis)
