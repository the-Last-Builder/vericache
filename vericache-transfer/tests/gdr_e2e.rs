//! Cross-process GPUDirect RDMA between distinct CUDA allocations on two GPUs.
#[allow(dead_code)] // The shared fixture also supports host-only lifecycle tests.
#[path = "common/rdma.rs"]
mod common;
#[path = "common/data_plane.rs"]
mod data_plane;

#[test]
fn physical_gdr_two_gpu_read_round_trip() {
    data_plane::run(
        "physical_gdr_two_gpu_read_round_trip",
        data_plane::Mode::GdrSingle,
    );
}

#[test]
fn physical_gdr_batched_multi_sge_pull() {
    data_plane::run(
        "physical_gdr_batched_multi_sge_pull",
        data_plane::Mode::GdrBatch,
    );
}
