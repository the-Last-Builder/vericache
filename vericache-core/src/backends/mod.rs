#[cfg(verus_only)]
use crate::frame::*;
use crate::{kernel::*, provenance::*, spec::*, timeline::*};
#[cfg(verus_only)]
use foundation::tracked::{closure_from, knowledge_of, untouched};
use transfer_core::{context::*, domain::*};
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::tokens::InstanceId;

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod device;
pub mod host;
#[cfg(feature = "nvme")]
pub mod nvme;
#[cfg(feature = "rdma")]
pub mod rdma;
pub mod traits;

#[cfg(feature = "cuda")]
pub use device::CudaAttachment;
pub use device::{
    DeviceAttachment, DeviceBackend, DeviceLease, DeviceSlice, DeviceStream, NoDevice,
};
pub use traits::HostSlabBackend;

#[cfg(verus_only)]
pub use vericache_kernel::compose;
#[cfg(verus_only)]
pub use vericache_kernel::compose::*;
pub use vericache_kernel::spi::*;
