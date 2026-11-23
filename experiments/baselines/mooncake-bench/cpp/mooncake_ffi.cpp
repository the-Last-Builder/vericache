#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

// Binary-exact declarations matching Mooncake 0.3.14 (libmooncake_pg.so, GCC libstdc++ ABI).
namespace mooncake {

class Status {
public:
    enum class Code : uint16_t { kOk = 0 };
    Code code_;
    const char* msg_;

    Status() : code_(Code::kOk), msg_(nullptr) {}
    ~Status() {
        if (msg_) {
            delete[] msg_;
        }
    }
    Status(const Status&) = delete;
    Status& operator=(const Status&) = delete;
    Status(Status&& o) noexcept : code_(o.code_), msg_(o.msg_) {
        o.code_ = Code::kOk;
        o.msg_ = nullptr;
    }
    bool ok() const { return code_ == Code::kOk; }
    uint16_t code_u16() const { return static_cast<uint16_t>(code_); }
    const char* message() const { return msg_ ? msg_ : ""; }
};
static_assert(sizeof(Status) == 16, "mooncake::Status ABI size mismatch");

class Transport {
public:
    using SegmentID = uint64_t;
    using BatchID = uint64_t;

    struct TransferRequest {
        enum OpCode : int32_t { READ = 0, WRITE = 1 };
        OpCode opcode = READ;       // 0x00
        int32_t _pad0 = 0;          // 0x04
        void* source = nullptr;     // 0x08
        SegmentID target_id = 0;    // 0x10
        uint64_t target_offset = 0; // 0x18
        size_t length = 0;          // 0x20
        int32_t flags = 0;          // 0x28
        int32_t _pad1 = 0;          // 0x2c
        uint64_t _rsv0 = 0;         // 0x30
        uint64_t _rsv1 = 0;         // 0x38
        int32_t num_retries = 1;    // 0x40
        int32_t local_buf_idx = -1; // 0x44
    };
    static_assert(sizeof(TransferRequest) == 0x48, "TransferRequest ABI size mismatch");

    enum TransferStatusEnum : int32_t {
        WAITING = 0,
        PENDING = 1,
        INVALID = 2,
        CANNOT_SEND = 3,
        COMPLETED = 4,
        TIMEOUT = 5,
        FAILED = 6,
    };

    struct TransferStatus {
        TransferStatusEnum s = WAITING;
        int32_t _pad = 0;
        size_t transferred_bytes = 0;
    };
    static_assert(sizeof(TransferStatus) == 16, "TransferStatus ABI size mismatch");
};

class TransferEngine {
    alignas(8) uint8_t opaque_[0xa8];

public:
    explicit TransferEngine(bool auto_discover = false);
    TransferEngine(bool auto_discover, const std::vector<std::string>& filter);
    ~TransferEngine();

    int init(const std::string& metadata_conn_string,
             const std::string& local_server_name,
             const std::string& ip_or_host_name = "",
             uint64_t rpc_port = 12345);
    int getRpcPort();
    Transport::SegmentID openSegment(const std::string& segment_name);
    int closeSegment(Transport::SegmentID handle);
    int registerLocalMemory(void* addr,
                            size_t length,
                            const std::string& location,
                            bool remote_accessible = true,
                            bool update_metadata = true);
    int unregisterLocalMemory(void* addr, bool update_metadata = true);
    Transport::BatchID allocateBatchID(size_t batch_size);
    Status freeBatchID(Transport::BatchID batch_id);
    Status submitTransfer(Transport::BatchID batch_id,
                          const std::vector<Transport::TransferRequest>& entries);
    Status getTransferStatus(Transport::BatchID batch_id,
                             size_t task_id,
                             Transport::TransferStatus& status);
    Status getBatchTransferStatus(Transport::BatchID batch_id,
                                  Transport::TransferStatus& status);
};
static_assert(sizeof(TransferEngine) == 0xa8, "TransferEngine ABI size mismatch");

} // namespace mooncake

struct McTeHandle {
    mooncake::TransferEngine* te;
};

extern "C" {

// =========================================================================
// Mooncake TransferEngine C FFI
// =========================================================================

McTeHandle* mc_te_create(const char* rdma_dev, int* out_rpc_port) {
    // Keep glog quiet on info messages unless caller explicitly set GLOG_minloglevel.
    setenv("GLOG_minloglevel", "2", 0);

    std::vector<std::string> filters;
    if (rdma_dev != nullptr && rdma_dev[0] != '\0') {
        filters.emplace_back(rdma_dev);
    }
    auto* te = new mooncake::TransferEngine(true, filters);
    int rc = te->init("P2PHANDSHAKE", "127.0.0.1:0", "", 0);
    if (rc != 0) {
        fprintf(stderr, "[mooncake-ffi] TransferEngine::init failed rc=%d (dev=%s)\n",
                rc, rdma_dev ? rdma_dev : "");
        delete te;
        return nullptr;
    }
    if (out_rpc_port != nullptr) {
        *out_rpc_port = te->getRpcPort();
    }
    auto* h = new McTeHandle{te};
    return h;
}

void mc_te_destroy(McTeHandle* h) {
    if (!h) return;
    delete h->te;
    delete h;
}

int mc_te_register_memory(McTeHandle* h, void* addr, size_t len) {
    if (!h || !h->te) return -1;
    return h->te->registerLocalMemory(addr, len, "*", true, true);
}

int mc_te_unregister_memory(McTeHandle* h, void* addr) {
    if (!h || !h->te) return -1;
    return h->te->unregisterLocalMemory(addr, true);
}

int64_t mc_te_open_segment(McTeHandle* h, const char* seg_name) {
    if (!h || !h->te || !seg_name) return -1;
    auto seg_id = h->te->openSegment(std::string(seg_name));
    return static_cast<int64_t>(seg_id);
}

int mc_te_close_segment(McTeHandle* h, uint64_t seg_id) {
    if (!h || !h->te) return -1;
    return h->te->closeSegment(seg_id);
}

int mc_te_sync_read(McTeHandle* h,
                    uint64_t seg_id,
                    void* local_ptr,
                    uint64_t remote_ptr,
                    size_t len) {
    if (!h || !h->te) return -1;
    auto* te = h->te;
    auto batch_id = te->allocateBatchID(1);
    if (batch_id == 0) return -2;

    std::vector<mooncake::Transport::TransferRequest> reqs(1);
    reqs[0].opcode = mooncake::Transport::TransferRequest::READ;
    reqs[0].source = local_ptr;
    reqs[0].target_id = seg_id;
    reqs[0].target_offset = remote_ptr;
    reqs[0].length = len;

    auto st = te->submitTransfer(batch_id, reqs);
    if (!st.ok()) {
        fprintf(stderr, "[mooncake-ffi] submitTransfer failed code=%u msg=%s\n",
                st.code_u16(), st.message());
        te->freeBatchID(batch_id);
        return -3;
    }

    int rc = 0;
    mooncake::Transport::TransferStatus ts;
    while (true) {
        auto st2 = te->getBatchTransferStatus(batch_id, ts);
        if (!st2.ok()) {
            fprintf(stderr, "[mooncake-ffi] getBatchTransferStatus failed code=%u msg=%s\n",
                    st2.code_u16(), st2.message());
            rc = -4;
            break;
        }
        if (ts.s == mooncake::Transport::COMPLETED) {
            rc = 0;
            break;
        }
        if (ts.s == mooncake::Transport::FAILED ||
            ts.s == mooncake::Transport::TIMEOUT ||
            ts.s == mooncake::Transport::INVALID ||
            ts.s == mooncake::Transport::CANNOT_SEND) {
            fprintf(stderr, "[mooncake-ffi] transfer terminated with status=%d\n",
                    static_cast<int>(ts.s));
            rc = -5;
            break;
        }
    }
    te->freeBatchID(batch_id);
    return rc;
}

int mc_te_batch_sync_read(McTeHandle* h,
                          uint64_t seg_id,
                          void* const* local_ptrs,
                          const uint64_t* remote_ptrs,
                          const size_t* lens,
                          size_t count) {
    if (!h || !h->te || count == 0) return -1;
    auto* te = h->te;
    auto batch_id = te->allocateBatchID(count);
    if (batch_id == 0) return -2;

    std::vector<mooncake::Transport::TransferRequest> reqs(count);
    for (size_t i = 0; i < count; ++i) {
        reqs[i].opcode = mooncake::Transport::TransferRequest::READ;
        reqs[i].source = local_ptrs[i];
        reqs[i].target_id = seg_id;
        reqs[i].target_offset = remote_ptrs[i];
        reqs[i].length = lens[i];
    }

    auto st = te->submitTransfer(batch_id, reqs);
    if (!st.ok()) {
        fprintf(stderr, "[mooncake-ffi] batch submitTransfer failed code=%u msg=%s\n",
                st.code_u16(), st.message());
        te->freeBatchID(batch_id);
        return -3;
    }

    int rc = 0;
    mooncake::Transport::TransferStatus ts;
    while (true) {
        auto st2 = te->getBatchTransferStatus(batch_id, ts);
        if (!st2.ok()) {
            fprintf(stderr, "[mooncake-ffi] batch getBatchTransferStatus failed code=%u msg=%s\n",
                    st2.code_u16(), st2.message());
            rc = -4;
            break;
        }
        if (ts.s == mooncake::Transport::COMPLETED) {
            rc = 0;
            break;
        }
        if (ts.s == mooncake::Transport::FAILED ||
            ts.s == mooncake::Transport::TIMEOUT ||
            ts.s == mooncake::Transport::INVALID ||
            ts.s == mooncake::Transport::CANNOT_SEND) {
            fprintf(stderr, "[mooncake-ffi] batch transfer terminated with status=%d\n",
                    static_cast<int>(ts.s));
            rc = -5;
            break;
        }
    }
    te->freeBatchID(batch_id);
    return rc;
}

// =========================================================================
// CUDA Runtime FFI (libcudart)
// =========================================================================

int mc_cuda_set_device(int dev) {
    return static_cast<int>(cudaSetDevice(dev));
}

int mc_cuda_malloc(size_t bytes, void** out_ptr) {
    return static_cast<int>(cudaMalloc(out_ptr, bytes > 0 ? bytes : 1));
}

int mc_cuda_free(void* ptr) {
    return static_cast<int>(cudaFree(ptr));
}

// Allocate 4096-byte aligned zeroed host memory and pin it with cudaHostRegister
// (matching VeriCache's HostBackend + attach_pinned_host for identical RSS and O_DIRECT alignment).
int mc_cuda_pinned_alloc(size_t bytes, void** out_ptr) {
    size_t alloc_sz = bytes > 0 ? bytes : 4096;
    void* p = nullptr;
    int rc = posix_memalign(&p, 4096, alloc_sz);
    if (rc != 0 || p == nullptr) {
        return -1;
    }
    std::memset(p, 0, alloc_sz);
    cudaError_t err = cudaHostRegister(p, alloc_sz, cudaHostRegisterPortable);
    if (err != cudaSuccess) {
        free(p);
        return static_cast<int>(err);
    }
    *out_ptr = p;
    return 0;
}

int mc_cuda_pinned_free(void* ptr) {
    if (!ptr) return 0;
    cudaError_t err = cudaHostUnregister(ptr);
    free(ptr);
    return static_cast<int>(err);
}

int mc_cuda_memset(void* ptr, int val, size_t bytes) {
    return static_cast<int>(cudaMemset(ptr, val, bytes));
}

int mc_cuda_stream_create(void** out_stream) {
    cudaStream_t s = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    *out_stream = reinterpret_cast<void*>(s);
    return static_cast<int>(err);
}

int mc_cuda_stream_destroy(void* stream) {
    if (!stream) return 0;
    return static_cast<int>(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
}

int mc_cuda_stream_sync(void* stream) {
    return static_cast<int>(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

int mc_cuda_device_sync() {
    return static_cast<int>(cudaDeviceSynchronize());
}

int mc_cuda_memcpy_async_d2d(void* dst, const void* src, size_t bytes, void* stream) {
    return static_cast<int>(cudaMemcpyAsync(
        dst, src, bytes, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream)));
}

int mc_cuda_memcpy_async_h2d(void* dst, const void* src, size_t bytes, void* stream) {
    return static_cast<int>(cudaMemcpyAsync(
        dst, src, bytes, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
}

int mc_cuda_memcpy_async_d2h(void* dst, const void* src, size_t bytes, void* stream) {
    return static_cast<int>(cudaMemcpyAsync(
        dst, src, bytes, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)));
}

// =========================================================================
// POSIX O_DIRECT NVMe I/O FFI
// =========================================================================

int mc_odirect_open_unlinked(const char* path, uint64_t size_bytes) {
    int fd = open(path, O_CREAT | O_RDWR | O_DIRECT | O_TRUNC, 0600);
    if (fd < 0) return -errno;
    if (ftruncate(fd, static_cast<off_t>(size_bytes)) != 0) {
        int e = errno;
        close(fd);
        unlink(path);
        return -e;
    }
    // Unlink immediately so Linux guarantees 0 leaked scratch files even on SIGKILL.
    unlink(path);
    return fd;
}

int mc_odirect_pread(int fd, void* buf, size_t len, uint64_t offset) {
    auto* p = static_cast<uint8_t*>(buf);
    size_t done = 0;
    while (done < len) {
        ssize_t n = pread(fd, p + done, len - done, static_cast<off_t>(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return -errno;
        }
        if (n == 0) return -EIO;
        done += static_cast<size_t>(n);
    }
    return 0;
}

int mc_odirect_pwrite(int fd, const void* buf, size_t len, uint64_t offset) {
    const auto* p = static_cast<const uint8_t*>(buf);
    size_t done = 0;
    while (done < len) {
        ssize_t n = pwrite(fd, p + done, len - done, static_cast<off_t>(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return -errno;
        }
        if (n == 0) return -EIO;
        done += static_cast<size_t>(n);
    }
    return 0;
}

int mc_odirect_close(int fd) {
    if (fd < 0) return 0;
    return close(fd) == 0 ? 0 : -errno;
}

} // extern "C"
