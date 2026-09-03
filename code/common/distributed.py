"""Minimal torch.distributed helpers for embarrassingly parallel experiment shards."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ShardPlan:
    shard_index: int
    num_shards: int
    rank: int
    world_size: int
    local_rank: int


_STATE: ShardPlan | None = None


def launched_with_torchrun() -> bool:
    return "LOCAL_RANK" in os.environ or "RANK" in os.environ


def init_distributed(backend: str | None = None) -> ShardPlan:
    global _STATE
    if _STATE is not None:
        return _STATE
    if not launched_with_torchrun():
        raise RuntimeError("distributed mode requires torchrun (LOCAL_RANK/RANK not set)")
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        backend = backend or "nccl"
    else:
        local_rank = 0
        backend = backend or "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    _STATE = ShardPlan(
        shard_index=rank,
        num_shards=world_size,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
    return _STATE


def current_shard_plan() -> ShardPlan | None:
    return _STATE


def resolve_sharding(
    shard_index: int,
    num_shards: int,
    *,
    distributed: bool,
) -> tuple[int, int, int, int]:
    if not distributed:
        return shard_index, num_shards, 0, 1
    plan = init_distributed()
    return (
        shard_index * plan.world_size + plan.rank,
        num_shards * plan.world_size,
        plan.rank,
        plan.world_size,
    )


def is_main_process() -> bool:
    if _STATE is None:
        return True
    return _STATE.rank == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    global _STATE
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    _STATE = None
