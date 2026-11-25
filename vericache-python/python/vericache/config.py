"""Settings accepted by the native bridge and the vLLM connector."""

from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class VeriCacheConfig:
    num_shards: int = 4
    host_frames_per_shard: int = 128
    nvme_frames_per_shard: int = 0
    frame_size: int = 65536
    nvme_path: str | Path | None = None
    prefer_cuda_pinned: bool = True
    lib_path: str | Path | None = None
    model_tag: str = ""
    block_axis: int | None = None
    io_workers: int = 2
    staging_pool_size: int = 8
    enable_async_pipeline: bool = True
    use_zero_copy_load: bool = True
    ipc_endpoint: str | None = None
    tp_rank: int | None = None
    world_size: int | None = None
    # Scheme A distributed cross-instance Type-2B MW RDMA settings (Stage 2)
    distributed_mode: bool = False
    node_id: int = 0
    instance_id: int | None = None
    rdma_device: str | None = None
    rdma_gid_index: int = 3
    rdma_windows_per_peer: int = 16
    peer_fetch_enabled: bool = True
    peers: tuple[dict, ...] = ()
    global_directory_endpoint: str | None = None

    def __post_init__(self):
        if isinstance(self.peers, list):
            object.__setattr__(self, "peers", tuple(self.peers))
        for name in (
            "num_shards",
            "host_frames_per_shard",
            "frame_size",
            "staging_pool_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.nvme_frames_per_shard) is not int
            or self.nvme_frames_per_shard < 0
        ):
            raise ValueError("nvme_frames_per_shard must be a nonnegative integer")
        if self.nvme_frames_per_shard and not self.nvme_path:
            raise ValueError(
                "nvme_path is required when nvme_frames_per_shard is positive"
            )
        for name in (
            "prefer_cuda_pinned",
            "enable_async_pipeline",
            "use_zero_copy_load",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if type(self.io_workers) is not int or self.io_workers < 0:
            raise ValueError("io_workers must be a nonnegative integer")
        if self.ipc_endpoint is not None and not isinstance(self.ipc_endpoint, str):
            raise ValueError("ipc_endpoint must be a string or None")
        if self.world_size is not None and (
            type(self.world_size) is not int or self.world_size < 1
        ):
            raise ValueError("world_size must be a positive integer")
        if self.tp_rank is not None and (
            type(self.tp_rank) is not int
            or self.tp_rank < 0
            or (self.world_size is not None and self.tp_rank >= self.world_size)
        ):
            raise ValueError("tp_rank must be a valid nonnegative rank")
        if not isinstance(self.model_tag, str):
            raise ValueError("model_tag must be a string")
        if self.block_axis is not None and type(self.block_axis) is not int:
            raise ValueError("block_axis must be an integer or None")

    @classmethod
    def from_vllm_config(cls, vllm_config):
        transfer = getattr(vllm_config, "kv_transfer_config", None)
        extra = getattr(transfer, "kv_connector_extra_config", None) or {}
        return cls(
            **{
                field.name: extra[field.name]
                for field in fields(cls)
                if field.name in extra
            }
        )
