#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64
#include <errno.h>
#include <fcntl.h>
#include <linux/fs.h>
#include <linux/io_uring.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

static int named_temp(const char *dir) {
    static _Atomic unsigned long counter;
    int parent = open(dir, O_DIRECTORY | O_PATH | O_CLOEXEC);
    if (parent < 0) return -1;
    int fd;
    char name[80];
    do {
        unsigned long seq = atomic_fetch_add_explicit(&counter, 1, memory_order_relaxed);
        snprintf(name, sizeof(name), ".vericache-%ld-%lu", (long)getpid(), seq);
        fd = openat(parent, name, O_CREAT | O_EXCL | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
    } while (fd < 0 && errno == EEXIST);
    int error = errno;
    if (fd >= 0 && unlinkat(parent, name, 0) < 0) {
        error = errno;
        close(fd);
        fd = -1;
    }
    close(parent);
    errno = error;
    return fd;
}

static int verify_dio_align(int fd) {
#ifdef STATX_DIOALIGN
    struct statx stx = {0};
    if (statx(fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &stx) == 0 &&
        (stx.stx_mask & STATX_DIOALIGN)) {
        if ((stx.stx_dio_mem_align != 0 &&
             (stx.stx_dio_mem_align > 4096 || 4096 % stx.stx_dio_mem_align != 0)) ||
            (stx.stx_dio_offset_align != 0 &&
             (stx.stx_dio_offset_align > 4096 || 4096 % stx.stx_dio_offset_align != 0))) {
            return -EINVAL;
        }
    }
#endif
    return 0;
}

static int ensure_regular_size(int fd, uint64_t len) {
    if (fallocate(fd, 0, 0, (off_t)len) == 0 ||
        (errno == EOPNOTSUPP && ftruncate(fd, (off_t)len) == 0)) return 0;
    return -errno;
}

int vc_nvme_create(const char *dir, uint64_t len) {
    if (len == 0) return -EINVAL;
    if (len > INT64_MAX) return -EFBIG;
    int fd = open(dir, O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
    if (fd < 0 && (errno == EOPNOTSUPP || errno == EISDIR || errno == EINVAL))
        fd = named_temp(dir);
    if (fd < 0) return -errno;
    int rc = verify_dio_align(fd);
    if (rc == 0) rc = ensure_regular_size(fd, len);
    if (rc < 0) {
        close(fd);
        return rc;
    }
    return fd;
}

int vc_nvme_open_path(const char *path, uint64_t len, int create_if_missing) {
    if (!path || len == 0) return -EINVAL;
    if (len > INT64_MAX) return -EFBIG;
    int flags = O_RDWR | O_DIRECT | O_CLOEXEC | (create_if_missing ? O_CREAT : 0);
    int fd = open(path, flags, 0600);
    if (fd < 0) return -errno;
    int rc = verify_dio_align(fd);
    if (rc < 0) {
        close(fd);
        return rc;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        int error = errno;
        close(fd);
        return -error;
    }
    if (S_ISBLK(st.st_mode)) {
        uint64_t dev_bytes = 0;
        if (ioctl(fd, BLKGETSIZE64, &dev_bytes) < 0) {
            int error = errno;
            close(fd);
            return -error;
        }
        if (dev_bytes < len) {
            close(fd);
            return -ENOSPC;
        }
        return fd;
    }
    if (S_ISREG(st.st_mode)) {
        if (st.st_size >= 0 && (uint64_t)st.st_size >= len) return fd;
        rc = ensure_regular_size(fd, len);
        if (rc < 0) {
            close(fd);
            return rc;
        }
        return fd;
    }
    close(fd);
    return -EINVAL;
}

// The CQE only echoes one word. Its cookie retains the full caller ID and
// requested length until reap, even after the kernel recycles the SQE slot.
struct cookie {
    uint64_t id;
    union { unsigned len, next; };
};

struct ring {
    int fd;
    char *sq, *cq;
    struct io_uring_sqe *entries;
    size_t sq_size, cq_size, entries_size;
    struct io_uring_params p;
    unsigned sq_mask, cq_mask;
    unsigned tail, free_cookie;
    struct cookie cookies[];
};

void vc_uring_destroy(struct ring *r) {
    if (r->entries != MAP_FAILED) munmap(r->entries, r->entries_size);
    if (r->cq != MAP_FAILED) munmap(r->cq, r->cq_size);
    if (r->sq != MAP_FAILED) munmap(r->sq, r->sq_size);
    close(r->fd);
    free(r);
}

int vc_uring_new(unsigned depth, struct ring **out, unsigned *sq_count, unsigned *cq_count) {
    struct io_uring_params p = {0};
    int fd = syscall(__NR_io_uring_setup, depth, &p);
    if (fd < 0) return errno;
    if (!p.sq_entries || p.cq_entries < p.sq_entries) {
        close(fd);
        return EINVAL;
    }
    struct ring *r = calloc(1, sizeof(*r) + (size_t)p.cq_entries * sizeof(struct cookie));
    if (!r) {
        close(fd);
        return ENOMEM;
    }
    int error;
    r->p = p;
    r->fd = fd;
    r->sq_size = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    r->cq_size = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
    r->entries_size = p.sq_entries * sizeof(struct io_uring_sqe);
    r->sq = r->cq = MAP_FAILED;
    r->entries = MAP_FAILED;
    r->sq = mmap(NULL, r->sq_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, IORING_OFF_SQ_RING);
    if (r->sq == MAP_FAILED) goto fail;
    r->cq = mmap(NULL, r->cq_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, IORING_OFF_CQ_RING);
    if (r->cq == MAP_FAILED) goto fail;
    r->entries = mmap(NULL, r->entries_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, IORING_OFF_SQES);
    if (r->entries == MAP_FAILED) goto fail;
    r->sq_mask = *(unsigned *)(r->sq + p.sq_off.ring_mask);
    r->cq_mask = *(unsigned *)(r->cq + p.cq_off.ring_mask);
    for (unsigned i = 0; i < p.cq_entries; ++i) r->cookies[i].next = i + 1;
    *out = r;
    *sq_count = p.sq_entries;
    *cq_count = p.cq_entries;
    return 0;
fail:
    error = errno;
    vc_uring_destroy(r);
    return error;
}

static unsigned acquire(char *addr) {
    return atomic_load_explicit((_Atomic unsigned *)addr, memory_order_acquire);
}

static void publish(char *addr, unsigned value) {
    atomic_store_explicit((_Atomic unsigned *)addr, value, memory_order_release);
}

int vc_uring_stage_io(struct ring *r, int fd, uint64_t offset, void *buf,
                      unsigned len, uint64_t id, int write) {
    unsigned head = acquire(r->sq + r->p.sq_off.head);
    if (r->tail - head >= r->p.sq_entries || r->free_cookie == r->p.cq_entries)
        return EAGAIN;
    unsigned slot = r->tail & r->sq_mask;
    unsigned index = r->free_cookie;
    struct cookie *c = &r->cookies[index];
    r->free_cookie = c->next;
    c->id = id;
    c->len = len;
    r->entries[slot] = (struct io_uring_sqe){
        .opcode = write ? IORING_OP_WRITE : IORING_OP_READ, .fd = fd,
        .off = offset, .addr = (uintptr_t)buf, .len = len, .user_data = index
    };
    ((unsigned *)(r->sq + r->p.sq_off.array))[slot] = slot;
    ++r->tail;
    return 0;
}

int vc_uring_submit(struct ring *r, unsigned count) {
    publish(r->sq + r->p.sq_off.tail, r->tail);
    int n;
    do {
        n = syscall(__NR_io_uring_enter, r->fd, count, 0, 0, NULL, 0);
    } while (n < 0 && errno == EINTR);
    return n < 0 ? -errno : n;
}

int vc_uring_poll(struct ring *r, uint64_t *id, unsigned *len, int *res) {
    unsigned head = acquire(r->cq + r->p.cq_off.head);
    if (head == acquire(r->cq + r->p.cq_off.tail)) return 0;
    struct io_uring_cqe *entries = (void *)(r->cq + r->p.cq_off.cqes);
    struct io_uring_cqe cqe = entries[head & r->cq_mask];
    struct cookie *c = &r->cookies[cqe.user_data];
    *id = c->id;
    *len = c->len;
    *res = cqe.res;
    c->next = r->free_cookie;
    r->free_cookie = cqe.user_data;
    publish(r->cq + r->p.cq_off.head, head + 1);
    return 1;
}

int vc_uring_wait(struct ring *r) {
    int n;
    do {
        n = syscall(__NR_io_uring_enter, r->fd, 0, 1, IORING_ENTER_GETEVENTS, NULL, 0);
    } while (n < 0 && errno == EINTR);
    return n < 0 ? errno : 0;
}
