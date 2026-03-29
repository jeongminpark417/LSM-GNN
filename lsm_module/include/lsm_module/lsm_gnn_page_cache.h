#ifndef LSM_MODULE_LSM_GNN_PAGE_CACHE_H
#define LSM_MODULE_LSM_GNN_PAGE_CACHE_H

#include <cstddef>
#include <cstdint>

/*
 * LSM-GNN facade: extended page cache in lsm_gnn_embedded_page_cache.h (PVP path,
 * next_reuse / reuse_chunk, wb_find_slot*). Include this from LSM_NVMe instead of
 * <page_cache.h> so the module does not depend on a patched BAM tree.
 *
 * When page_cache_t::is_pvp is true, device eviction can stage to pinned queues;
 * call page_cache_t::bind_pinned_eviction_staging after pinned_eviction_buffer_init,
 * range_t::sync_embedded_page_cache_from, then array_t::resync_device_range_from so
 * kernels using array_d_t::d_ranges see the same embedded page_cache_d_t.
 */
#include "lsm_gnn_embedded_page_cache.h"

/** Sentinel for “no reuse hint” (matches embedded `no_reuse` macro). */
constexpr uint64_t lsm_gnn_next_reuse_unused_v() noexcept {
    return 0xFFFF000000000000ULL;
}

using lsm_gnn_data_page_t = data_page_t;
using lsm_gnn_pages_t = pages_t;
using lsm_gnn_page_cache_d_t = page_cache_d_t;
using lsm_gnn_page_cache_t = page_cache_t;

template <typename T>
using lsm_gnn_returned_cache_page_t = returned_cache_page_t<T>;

template <typename T, size_t n = 32,
          simt::thread_scope _scope = simt::thread_scope_device, size_t loc = GLOBAL_>
using lsm_gnn_tlb = tlb<T, n, _scope, loc>;

template <typename T, size_t n = 32,
          simt::thread_scope _scope = simt::thread_scope_device, size_t loc = GLOBAL_>
using lsm_gnn_bam_ptr_tlb = bam_ptr_tlb<T, n, _scope, loc>;

template <typename T>
using lsm_gnn_bam_ptr = bam_ptr<T>;

template <typename T>
using lsm_gnn_range_d_t = range_d_t<T>;

template <typename T>
using lsm_gnn_range_t = range_t<T>;

template <typename T>
using lsm_gnn_array_d_t = array_d_t<T>;

template <typename T>
using lsm_gnn_array_t = array_t<T>;

__host__ inline void lsm_gnn_init_next_reuse(page_cache_t* pc) {
    if (!pc || pc->pdt.n_pages == 0 || pc->pdt.cache_pages == nullptr)
        return;
    const uint64_t n = pc->pdt.n_pages;
    const std::size_t off = offsetof(cache_page_t, next_reuse);
    uint64_t sent = lsm_gnn_next_reuse_unused_v();
    for (uint64_t i = 0; i < n; ++i) {
        char* row = reinterpret_cast<char*>(pc->pdt.cache_pages) + i * sizeof(cache_page_t);
        cuda_err_chk(cudaMemcpy(row + off, &sent, sizeof(sent), cudaMemcpyHostToDevice));
    }
}

namespace lsm_gnn {
namespace detail {

template <typename T>
__forceinline__ __device__ bool logical_page_has_cache_line(range_d_t<T>* r, uint64_t page,
                                                            uint64_t& page_trans) {
    uint32_t read_state = r->pages[page].state.load(simt::memory_order_relaxed);
    uint32_t st = (read_state >> (CNT_SHIFT + 1)) & 0x03u;
    if (st == V_NB) {
        page_trans = r->pages[page].offset;
        return true;
    }
    return false;
}

}  // namespace detail

/** Defaults when init passes 0 (use same geometry as page_cache_t PVP ctor). */
inline constexpr uint32_t default_num_pvp_buffers() noexcept { return 256; }
inline constexpr uint32_t default_pvp_queue_depth() noexcept { return 32; }

/**
 * Host-pinned, GPU-mapped eviction staging (ring: cache_page_size * num_pvp_buffers
 * * pvp_queue_depth bytes).
 */
template <typename T>
struct pinned_eviction_buffer {
    uint32_t num_pvp_buffers = 0;
    uint32_t pvp_queue_depth = 0;
    uint32_t* pvp_queue_counter = nullptr;
    uint64_t* pvp_id_array = nullptr;
    T* host_queue = nullptr;
    T* device_queue = nullptr;
    uint64_t* host_id_storage = nullptr;

    __host__ bool valid() const {
        return host_queue != nullptr && pvp_queue_counter != nullptr;
    }
};

template <typename T>
__host__ inline void pinned_eviction_buffer_destroy(pinned_eviction_buffer<T>& b) {
    if (b.pvp_queue_counter != nullptr) {
        cuda_err_chk(cudaFree(b.pvp_queue_counter));
        b.pvp_queue_counter = nullptr;
    }
    if (b.host_queue != nullptr) {
        cuda_err_chk(cudaFreeHost(b.host_queue));
        b.host_queue = nullptr;
        b.device_queue = nullptr;
    }
    if (b.host_id_storage != nullptr) {
        cuda_err_chk(cudaFreeHost(b.host_id_storage));
        b.host_id_storage = nullptr;
        b.pvp_id_array = nullptr;
    }
    b.num_pvp_buffers = 0;
    b.pvp_queue_depth = 0;
}

template <typename T>
__host__ inline void pinned_eviction_buffer_init(pinned_eviction_buffer<T>& b,
                                                 std::size_t cache_page_size,
                                                 uint32_t num_pvp_buffers,
                                                 uint32_t pvp_queue_depth) {
    pinned_eviction_buffer_destroy(b);
    b.num_pvp_buffers = num_pvp_buffers;
    b.pvp_queue_depth = pvp_queue_depth;
    cuda_err_chk(cudaMalloc(&b.pvp_queue_counter, sizeof(uint32_t) * num_pvp_buffers));
    cuda_err_chk(cudaMemset(b.pvp_queue_counter, 0, sizeof(uint32_t) * num_pvp_buffers));
    const std::size_t queue_bytes = cache_page_size * static_cast<std::size_t>(num_pvp_buffers) *
                                    static_cast<std::size_t>(pvp_queue_depth);
    cuda_err_chk(cudaHostAlloc(reinterpret_cast<void**>(&b.host_queue), queue_bytes,
                               cudaHostAllocMapped));
    cudaHostGetDevicePointer(reinterpret_cast<void**>(&b.device_queue),
                             reinterpret_cast<void*>(b.host_queue), 0);
    const std::size_t id_bytes = sizeof(uint64_t) * static_cast<std::size_t>(num_pvp_buffers) *
                                 static_cast<std::size_t>(pvp_queue_depth);
    cuda_err_chk(cudaHostAlloc(reinterpret_cast<void**>(&b.host_id_storage), id_bytes,
                               cudaHostAllocMapped));
    cudaHostGetDevicePointer(reinterpret_cast<void**>(&b.pvp_id_array),
                             reinterpret_cast<void*>(b.host_id_storage), 0);
}

/**
 * PVP fetch: same indexing as plain read_feature, but cache fills use the eviction
 * path that can copy victims into the pinned ring (page_cache_t::is_pvp on device).
 */
template <typename T>
__global__ void read_feature_pvp_kernel(array_d_t<T>* dr, T* out_tensor_ptr,
                                        int64_t* index_ptr, int dim, int64_t num_idx,
                                        int cache_dim, uint8_t time_step, uint32_t head_ptr) {
    uint64_t bid = blockIdx.x;
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int idx_idx = bid * num_warps + warp_id;
    if (idx_idx < num_idx) {
        wb_bam_ptr<T> ptr(dr);
        ptr.set_time(time_step, head_ptr);

        uint64_t row_index = index_ptr[idx_idx];
        uint64_t tid = threadIdx.x % 32;
        for (; tid < static_cast<uint64_t>(dim); tid += 32) {
            T temp = ptr[(row_index)*cache_dim + tid];
            out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
        }
    }
}

template <typename T>
__forceinline__ __device__ void update_prefetch_timestamp(lsm_gnn_array_d_t<T>* array,
                                                          uint64_t page,
                                                          uint32_t prefetch_timestamp,
                                                          uint64_t idx) {
    range_d_t<T>* r_ = array->d_ranges;
    uint64_t page_trans = 0;
    if (!detail::logical_page_has_cache_line(r_, page, page_trans))
        return;

    cache_page_t* line = r_->cache.get_cache_page(static_cast<uint32_t>(page_trans));
    uint64_t update_val = (static_cast<uint64_t>(prefetch_timestamp) << 48) | idx;
    atomicMin(reinterpret_cast<unsigned long long*>(&line->next_reuse),
              static_cast<unsigned long long>(update_val));
}

__forceinline__ __device__ uint64_t read_next_reuse(const cache_page_t* base_line) {
    return base_line->next_reuse;
}

inline constexpr uint64_t next_reuse_not_resident_value() noexcept {
    return ~0ULL;
}

inline constexpr uint64_t pack_prefetch_timestamp_idx(uint32_t prefetch_timestamp,
                                                      uint64_t idx) noexcept {
    return (static_cast<uint64_t>(prefetch_timestamp) << 48) |
           (idx & ((1ULL << 48) - 1ULL));
}

inline constexpr uint32_t unpack_prefetch_timestamp(uint64_t packed) noexcept {
    return static_cast<uint32_t>(packed >> 48);
}

inline constexpr uint64_t unpack_prefetch_idx(uint64_t packed) noexcept {
    return packed & ((1ULL << 48) - 1ULL);
}

}  // namespace lsm_gnn

#endif  // LSM_MODULE_LSM_GNN_PAGE_CACHE_H
