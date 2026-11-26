// Native verbs layouts stay in this translation unit. All protocol state,
// resource selection, WR tracking and completion handling live in Rust.
#include <infiniband/verbs.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct ibv_device *vc_device_at(struct ibv_device **devices, unsigned index) {
    return devices[index];
}

const char *vc_device_name(struct ibv_context *ctx) {
    return ibv_get_device_name(ctx->device);
}

static uint8_t atomic_limit(int n) {
    return n < 0 ? 0 : n > UINT8_MAX ? UINT8_MAX : (uint8_t)n;
}

int vc_device_attr(struct ibv_context *ctx, int *type2, uint32_t *depth,
                   uint32_t *sge, uint32_t *sge_rd, uint8_t *responder, uint8_t *initiator) {
    struct ibv_device_attr attr;
    int rc = ibv_query_device(ctx, &attr);
    if (rc) return rc;
    *type2 = !!(attr.device_cap_flags & IBV_DEVICE_MEM_WINDOW_TYPE_2B);
    *depth = attr.max_qp_wr > 0 ? (uint32_t)attr.max_qp_wr : 0;
    *sge = attr.max_sge > 0 ? (uint32_t)attr.max_sge : 0;
    *sge_rd = attr.max_sge_rd > 0 ? (uint32_t)attr.max_sge_rd : *sge;
    *responder = atomic_limit(attr.max_qp_rd_atom);
    *initiator = atomic_limit(attr.max_qp_init_rd_atom);
    return 0;
}

int vc_port(struct ibv_context *ctx, uint8_t port, uint16_t *lid,
            uint8_t *mtu, uint8_t *state, uint8_t *ethernet) {
    struct ibv_port_attr attr = {0};
    int rc = ibv_query_port(ctx, port, &attr);
    *lid = attr.lid;
    *mtu = attr.active_mtu;
    *state = attr.state;
    *ethernet = attr.link_layer == IBV_LINK_LAYER_ETHERNET;
    return rc;
}

int vc_gid(struct ibv_context *ctx, uint8_t port, int index, uint8_t *bytes) {
    if (index < 0) return EINVAL;
    struct ibv_gid_entry entry = {0};
    errno = 0;
    int rc = ibv_query_gid_ex(ctx, port, (uint32_t)index, &entry, 0);
    if (rc == 0) {
        if (entry.gid_type != IBV_GID_TYPE_ROCE_V2) return EINVAL;
        memcpy(bytes, entry.gid.raw, sizeof(entry.gid.raw));
        return 0;
    }
    int err = rc < 0 ? (errno ? errno : EIO) : rc;
    if (err != EOPNOTSUPP && err != ENOSYS) return err;
    union ibv_gid gid = {0};
    rc = ibv_query_gid(ctx, port, index, &gid);
    if (rc) return rc < 0 ? (errno ? errno : EIO) : rc;
    memcpy(bytes, gid.raw, sizeof(gid.raw));
    return 0;
}

struct ibv_mr *vc_register(struct ibv_pd *pd, void *addr, size_t len, int write, int bind) {
    return ibv_reg_mr(pd, addr, len, (write ? IBV_ACCESS_LOCAL_WRITE : 0) |
                                     (bind ? IBV_ACCESS_MW_BIND : 0));
}
uint32_t vc_lkey(struct ibv_mr *mr) { return mr->lkey; }
struct ibv_mw *vc_alloc_window(struct ibv_pd *pd) { return ibv_alloc_mw(pd, IBV_MW_TYPE_2); }
uint32_t vc_rkey(struct ibv_mw *mw) { return mw->rkey; }
int vc_dealloc_window(struct ibv_mw *mw) { return ibv_dealloc_mw(mw); }

struct ibv_qp *vc_create_qp(struct ibv_pd *pd, struct ibv_cq *cq, unsigned depth, unsigned max_send_sge) {
    struct ibv_qp_init_attr attr = {
        .send_cq = cq, .recv_cq = cq, .qp_type = IBV_QPT_RC,
        .cap = {.max_send_wr = depth, .max_recv_wr = 1, .max_send_sge = max_send_sge, .max_recv_sge = 1}
    };
    return ibv_create_qp(pd, &attr);
}
uint32_t vc_qpn(struct ibv_qp *qp) { return qp->qp_num; }

int vc_init(struct ibv_qp *qp, uint8_t port) {
    struct ibv_qp_attr attr = {.qp_state = IBV_QPS_INIT, .port_num = port,
        .pkey_index = 0, .qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE};
    return ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX | IBV_QP_ACCESS_FLAGS);
}

int vc_rtr(struct ibv_qp *qp, uint8_t port, uint32_t qpn, uint32_t psn, uint16_t lid,
           uint8_t mtu, uint8_t global, uint8_t index, const uint8_t *gid, uint8_t responder) {
    struct ibv_qp_attr attr = {.qp_state = IBV_QPS_RTR, .path_mtu = mtu,
        .dest_qp_num = qpn, .rq_psn = psn, .max_dest_rd_atomic = responder, .min_rnr_timer = 12,
        .ah_attr = {.dlid = lid, .port_num = port, .is_global = global,
                    .grh = {.sgid_index = index, .hop_limit = 64}}};
    memcpy(attr.ah_attr.grh.dgid.raw, gid, 16);
    return ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

int vc_rts(struct ibv_qp *qp, uint32_t psn, uint8_t initiator) {
    struct ibv_qp_attr attr = {.qp_state = IBV_QPS_RTS, .sq_psn = psn, .timeout = 14,
        .retry_cnt = 7, .rnr_retry = 7, .max_rd_atomic = initiator};
    return ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
        IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC);
}

int vc_error(struct ibv_qp *qp) {
    struct ibv_qp_attr attr = {.qp_state = IBV_QPS_ERR};
    return ibv_modify_qp(qp, &attr, IBV_QP_STATE);
}

int vc_reset(struct ibv_qp *qp) {
    struct ibv_qp_attr attr = {.qp_state = IBV_QPS_RESET};
    return ibv_modify_qp(qp, &attr, IBV_QP_STATE);
}

// wc.opcode is undefined on failure. Return the submitted kind through the
// opaque WR cookie instead. Most IDs fit inline. Wide IDs keep all 64 bits
// in an aligned cookie, released on post failure or when the CQE is polled.
enum work_kind { DATA, BIND_WINDOW, LOCAL_INVALIDATE, WIDE_ID };
struct wide_cookie {
    uint64_t id;
    uint32_t kind;
};

_Static_assert(_Alignof(struct wide_cookie) >= 4, "wide_cookie alignment");

static int pack_wr_id(uint64_t *id, enum work_kind kind) {
    if (*id <= (UINT64_MAX >> 2)) {
        *id = (*id << 2) | kind;
    } else {
        struct wide_cookie *wide = malloc(sizeof(*wide));
        if (!wide) return ENOMEM;
        *wide = (struct wide_cookie){.id = *id, .kind = kind};
        *id = (uintptr_t)wide | WIDE_ID;
    }
    return 0;
}

static void release_cookie(uint64_t id) {
    if ((id & 3u) == WIDE_ID) free((void *)(uintptr_t)(id & ~UINT64_C(3)));
}

static int post_signaled(struct ibv_qp *qp, struct ibv_send_wr *wr, enum work_kind kind) {
    struct ibv_send_wr *bad;
    int rc = pack_wr_id(&wr->wr_id, kind);
    if (rc) return rc;
    rc = ibv_post_send(qp, wr, &bad);
    if (rc) release_cookie(wr->wr_id);
    return rc;
}

struct vc_bind_desc {
    struct ibv_mw *mw;
    struct ibv_mr *mr;
    uint64_t addr, len, id;
    uint32_t rkey;
    int32_t write;
};
struct vc_inv_desc { uint64_t id; uint32_t rkey; };

static struct ibv_send_wr bind_wr(const struct vc_bind_desc *d) {
    return (struct ibv_send_wr){.wr_id = d->id, .opcode = IBV_WR_BIND_MW,
        .send_flags = IBV_SEND_SIGNALED, .bind_mw = {.mw = d->mw, .rkey = d->rkey,
        .bind_info = {.mr = d->mr, .addr = d->addr, .length = d->len,
        .mw_access_flags = IBV_ACCESS_REMOTE_READ | (d->write ? IBV_ACCESS_REMOTE_WRITE : 0)}}};
}

static struct ibv_send_wr invalidate_wr(const struct vc_inv_desc *d) {
    return (struct ibv_send_wr){.wr_id = d->id, .opcode = IBV_WR_LOCAL_INV,
        .send_flags = IBV_SEND_SIGNALED, .invalidate_rkey = d->rkey};
}

int vc_bind(struct ibv_qp *qp, struct ibv_mw *mw, struct ibv_mr *mr, uint64_t addr,
            uint64_t len, uint32_t rkey, int write, uint64_t id) {
    struct vc_bind_desc d = {.mw = mw, .mr = mr, .addr = addr, .len = len,
                             .id = id, .rkey = rkey, .write = write};
    struct ibv_send_wr wr = bind_wr(&d);
    return post_signaled(qp, &wr, BIND_WINDOW);
}
int vc_invalidate(struct ibv_qp *qp, uint32_t rkey, uint64_t id) {
    struct vc_inv_desc d = {.id = id, .rkey = rkey};
    struct ibv_send_wr wr = invalidate_wr(&d);
    return post_signaled(qp, &wr, LOCAL_INVALIDATE);
}

static int post_wr_list(struct ibv_qp *qp, struct ibv_send_wr *wrs, size_t count,
                        size_t *posted_out) {
    struct ibv_send_wr *bad = NULL;
    int rc = ibv_post_send(qp, wrs, &bad);
    if (!rc) {
        *posted_out = count;
    } else {
        // verbs reports the first unposted WR on failure.
        uintptr_t offset = (uintptr_t)bad - (uintptr_t)wrs;
        if (offset % sizeof(*wrs) || offset / sizeof(*wrs) >= count) abort();
        *posted_out = offset / sizeof(*wrs);
    }
    return rc;
}

static int post_control_batch(struct ibv_qp *qp, const void *descs, size_t count,
                               enum work_kind kind, size_t *posted_out) {
    struct ibv_send_wr local[32];
    *posted_out = 0;
    if (!count) return 0;
    struct ibv_send_wr *wrs = count <= 32 ? local : calloc(count, sizeof(*wrs));
    if (!wrs) return ENOMEM;
    size_t packed = 0;
    int rc = 0;
    while (packed < count) {
        wrs[packed] = kind == BIND_WINDOW
            ? bind_wr(&((const struct vc_bind_desc *)descs)[packed])
            : invalidate_wr(&((const struct vc_inv_desc *)descs)[packed]);
        rc = pack_wr_id(&wrs[packed].wr_id, kind);
        if (rc) break;
        ++packed;
        if (packed < count) wrs[packed - 1].next = &wrs[packed];
    }
    if (!rc) rc = post_wr_list(qp, wrs, count, posted_out);
    for (size_t i = *posted_out; i < packed; ++i) release_cookie(wrs[i].wr_id);
    if (wrs != local) free(wrs);
    return rc;
}

int vc_bind_batch(struct ibv_qp *qp, const struct vc_bind_desc *descs, size_t count,
                  size_t *posted_out) {
    return post_control_batch(qp, descs, count, BIND_WINDOW, posted_out);
}

int vc_invalidate_batch(struct ibv_qp *qp, const struct vc_inv_desc *descs, size_t count,
                        size_t *posted_out) {
    return post_control_batch(qp, descs, count, LOCAL_INVALIDATE, posted_out);
}
struct vc_sge { uint64_t addr; uint32_t len, lkey; };
struct vc_wr_desc {
    uint64_t id, remote;
    uint32_t rkey, num_sge;
    uint8_t read, fence;
};
struct vc_wc { uint64_t id; int32_t status; uint32_t kind; };

static struct ibv_send_wr data_wr(const struct vc_wr_desc *d, struct ibv_sge *native,
                                  const struct vc_sge *sges) {
    for (uint32_t i = 0; i < d->num_sge; ++i)
        native[i] = (struct ibv_sge){.addr = sges[i].addr, .length = sges[i].len, .lkey = sges[i].lkey};
    return (struct ibv_send_wr){.wr_id = d->id, .sg_list = native, .num_sge = (int)d->num_sge,
        .opcode = d->read ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE,
        .send_flags = IBV_SEND_SIGNALED | (d->fence ? IBV_SEND_FENCE : 0),
        .wr.rdma = {.remote_addr = d->remote, .rkey = d->rkey}};
}

int vc_transfer_sg(struct ibv_qp *qp, const struct vc_wr_desc *d, const struct vc_sge *sges) {
    struct ibv_sge small[32];
    if (!d->num_sge || d->num_sge > INT_MAX) return EINVAL;
    struct ibv_sge *native = d->num_sge <= 32 ? small : calloc(d->num_sge, sizeof(*native));
    if (!native) return ENOMEM;
    struct ibv_send_wr wr = data_wr(d, native, sges);
    int rc = post_signaled(qp, &wr, DATA);
    if (native != small) free(native);
    return rc;
}

int vc_transfer(struct ibv_qp *qp, uint64_t local, uint32_t len, uint32_t lkey,
                uint64_t remote, uint32_t rkey, int read, int fence, uint64_t id) {
    struct vc_wr_desc d = {.id = id, .remote = remote, .rkey = rkey, .num_sge = 1,
                           .read = !!read, .fence = !!fence};
    struct vc_sge local_sge = {.addr = local, .len = len, .lkey = lkey};
    struct ibv_sge native;
    struct ibv_send_wr wr = data_wr(&d, &native, &local_sge);
    return post_signaled(qp, &wr, DATA);
}

int vc_post_batch(struct ibv_qp *qp, const struct vc_wr_desc *reqs, size_t count,
                  const struct vc_sge *sges, size_t total_sges, size_t *posted_out) {
    struct ibv_send_wr local_wrs[32];
    struct ibv_sge local_sges[64];
    *posted_out = 0;
    if (!count) return 0;
    struct ibv_send_wr *wrs = count <= 32 ? local_wrs : calloc(count, sizeof(*wrs));
    struct ibv_sge *native = total_sges <= 64 ? local_sges : calloc(total_sges, sizeof(*native));
    int rc = ENOMEM;
    size_t packed = 0, offset = 0;
    if (!wrs || !native) goto cleanup;
    while (packed < count) {
        const struct vc_wr_desc *d = &reqs[packed];
        rc = EINVAL;
        if (!d->num_sge || d->num_sge > INT_MAX || d->num_sge > total_sges - offset) goto cleanup;
        wrs[packed] = data_wr(d, native + offset, sges + offset);
        rc = pack_wr_id(&wrs[packed].wr_id, DATA);
        if (rc) goto cleanup;
        ++packed;
        if (packed < count) wrs[packed - 1].next = &wrs[packed];
        offset += d->num_sge;
    }
    if (offset != total_sges) { rc = EINVAL; goto cleanup; }
    rc = post_wr_list(qp, wrs, count, posted_out);
cleanup:
    for (size_t i = *posted_out; i < packed; ++i) release_cookie(wrs[i].wr_id);
    if (native != local_sges) free(native);
    if (wrs != local_wrs) free(wrs);
    return rc;
}

_Static_assert(IBV_WC_SUCCESS == 0, "unexpected completion ABI");

static struct vc_wc unpack_wc(struct ibv_wc wc) {
    struct vc_wc result = {.id = wc.wr_id >> 2, .kind = wc.wr_id & 3u, .status = wc.status};
    if (result.kind == WIDE_ID) {
        struct wide_cookie *wide = (void *)(uintptr_t)(wc.wr_id & ~UINT64_C(3));
        result.id = wide->id;
        result.kind = wide->kind;
        free(wide);
    }
    return result;
}

// A poll can consume CQEs before the caller knows how much output to reserve.
// Keep those CQEs on the queue until either poll API takes them.
struct poll_buffer {
    int first, end;
    struct ibv_wc wcs[];
};

struct ibv_cq *vc_create_cq(struct ibv_context *ctx, int depth) {
    struct ibv_cq *cq = ibv_create_cq(ctx, depth, NULL, NULL, 0);
    if (!cq) return NULL;
    size_t capacity = (size_t)cq->cqe;
    struct poll_buffer *buf = NULL;
    if (capacity <= (SIZE_MAX - sizeof(*buf)) / sizeof(*buf->wcs))
        buf = calloc(1, sizeof(*buf) + capacity * sizeof(*buf->wcs));
    if (!buf) {
        ibv_destroy_cq(cq);
        errno = ENOMEM;
        return NULL;
    }
    cq->cq_context = buf;
    return cq;
}

int vc_destroy_cq(struct ibv_cq *cq) {
    struct poll_buffer *buf = cq->cq_context;
    int rc = ibv_destroy_cq(cq);
    if (!rc && buf) {
        for (int i = buf->first; i < buf->end; ++i) release_cookie(buf->wcs[i].wr_id);
        free(buf);
    }
    return rc;
}

int vc_poll_batch(struct ibv_cq *cq, int max_entries) {
    struct poll_buffer *buf = cq->cq_context;
    if (buf->first == buf->end) {
        int limit = max_entries < cq->cqe ? max_entries : cq->cqe;
        int n = ibv_poll_cq(cq, limit, buf->wcs);
        if (n <= 0) return n;
        buf->first = 0;
        buf->end = n;
    }
    int available = buf->end - buf->first;
    return max_entries < available ? max_entries : available;
}

void vc_take_batch(struct ibv_cq *cq, struct vc_wc *out, int count) {
    struct poll_buffer *buf = cq->cq_context;
    if (count < 0 || count > buf->end - buf->first) abort();
    for (int i = 0; i < count; ++i) out[i] = unpack_wc(buf->wcs[buf->first++]);
}

int vc_poll(struct ibv_cq *cq, uint64_t *id, int *status, uint32_t *kind) {
    int n = vc_poll_batch(cq, 1);
    if (n == 1) {
        struct vc_wc result;
        vc_take_batch(cq, &result, 1);
        *id = result.id;
        *status = result.status;
        *kind = result.kind;
    }
    return n;
}
