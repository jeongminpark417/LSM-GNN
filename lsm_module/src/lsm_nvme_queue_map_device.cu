/**
 * cuco + Thrust only — do not include BAM / lsm_gnn headers here (simt::std clashes with CCCL).
 *
 * Node reuse map (stored in LSM_NVMeQueueMapState::map):
 * - Each call clears the map and rebuilds from (node_id, batch_index) pairs.
 * - batch_index = which batch / PVP buffer the row belongs to (0, 1, 2, …).
 * - Single distinct batch index for a node (or same index repeated): value = INT32_MAX.
 * - Otherwise: value = second-smallest distinct batch_index (smallest is anchor); if all same index,
 *   value = INT32_MAX.
 *
 * index_map_add (see lsm_nvme_queue_map_index_map_add_impl): cuco insert_or_apply with pair
 * (key, INT32_MAX); new slots store INT32_MAX; existing payload INT32_MAX is replaced with
 * time_step; other payloads unchanged.
 *
 * index_map_remove: find + erase if payload is INT32_MAX or matches time_step. The static_map is
 * built with erased_key -2 (empty key remains -1); valid node ids must not use these sentinels.
 *
 * Payload sentinels: cuCollections ``empty_value`` for the *mapped* type must not equal the
 * semantic ``INT32_MAX`` stored for "single batch / unused" — otherwise device ``insert_or_apply``
 * calls ``wait_for_payload`` until payload != empty_value and spins forever on keys that already
 * hold INT32_MAX. We use INT32_MIN only as the cuco empty-payload pattern; user-visible values
 * remain INT32_MAX, batch indices, and time_step.
 */

#include <cuco/static_map.cuh>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

namespace {

// cuco empty **mapped** sentinel — must differ from semantic map value INT32_MAX (see file comment).
constexpr int32_t k_cuco_empty_payload_v = std::numeric_limits<int32_t>::min();
// Public / documented "unused reuse" payload (avoid std::numeric_limits in __global__).
constexpr int32_t k_map_semantic_int32_max = 2147483647;

}  // namespace

struct LSM_NVMeQueueMapState {
  using MapType =
      cuco::static_map<int64_t, int32_t, cuco::extent<std::size_t>,
                       cuda::thread_scope_device>;
  std::unique_ptr<MapType> map;
  std::size_t requested_capacity = 0;
};

using LSM_NVMeQueueMapInsertOrApplyRef =
    decltype(std::declval<const LSM_NVMeQueueMapState::MapType &>().ref(
        cuco::insert_or_apply));

using LSM_NVMeQueueMapRemoveRef =
    decltype(std::declval<const LSM_NVMeQueueMapState::MapType &>().ref(
        cuco::contains, cuco::find, cuco::erase));

namespace {

struct lsm_nvme_index_map_add_op {
  int32_t ts;
  __device__ void operator()(
      cuda::atomic_ref<int32_t, cuda::thread_scope_device> slot,
      int32_t /*pair_second*/) const noexcept {
    constexpr int32_t k_max = k_map_semantic_int32_max;
    int32_t const cur = slot.load(cuda::memory_order_relaxed);
    if (cur == k_max) {
      slot.store(ts, cuda::memory_order_relaxed);
    }
  }
};

}  // namespace

__global__ void lsm_nvme_index_map_add_kernel(LSM_NVMeQueueMapInsertOrApplyRef ref,
                                              int64_t const *__restrict__ nodes, int n,
                                              int32_t time_step_as_i32) {
  namespace cg = cooperative_groups;
  constexpr int CG = static_cast<int>(LSM_NVMeQueueMapInsertOrApplyRef::cg_size);
  auto tile = cg::tiled_partition<CG>(cg::this_thread_block());

  int const tile_global = (blockIdx.x * blockDim.x + threadIdx.x) / CG;
  if (tile_global >= n)
    return;

  int64_t const key = nodes[tile_global];
  if (key < 0)
    return;

  constexpr int32_t k_max = k_map_semantic_int32_max;
  ref.insert_or_apply(tile, cuco::pair<int64_t, int32_t>{key, k_max},
                      lsm_nvme_index_map_add_op{time_step_as_i32});
}

__global__ void lsm_nvme_index_map_remove_kernel(LSM_NVMeQueueMapRemoveRef ref,
                                                 int64_t const *__restrict__ nodes, int n,
                                                 int32_t time_step_as_i32) {
  namespace cg = cooperative_groups;
  constexpr int CG = static_cast<int>(LSM_NVMeQueueMapRemoveRef::cg_size);
  auto tile = cg::tiled_partition<CG>(cg::this_thread_block());

  int const tile_global = (blockIdx.x * blockDim.x + threadIdx.x) / CG;
  if (tile_global >= n)
    return;

  int64_t const key = nodes[tile_global];
  if (key < 0)
    return;

  if (!ref.contains(tile, key))
    return;

  constexpr int32_t k_max = k_map_semantic_int32_max;
  auto it = ref.find(tile, key);
  int32_t const v = it->second;
  if (v != k_max && v != time_step_as_i32)
    return;

  (void)ref.erase(tile, key);
}

namespace {

struct is_segment_head {
  cuco::pair<int64_t, int32_t> const *__restrict__ pairs;
  __host__ __device__ bool operator()(int i) const {
    return i == 0 || pairs[i].first != pairs[i - 1].first;
  }
};

__global__ void fill_node_reuse_map_entries_kernel(
    cuco::pair<int64_t, int32_t> const *__restrict__ sorted, int m,
    int const *__restrict__ head_starts, int nh,
    cuco::pair<int64_t, int32_t> *__restrict__ out) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nh)
    return;

  int const start = head_starts[t];
  int end = start + 1;
  while (end < m && sorted[end].first == sorted[start].first)
    ++end;

  constexpr int32_t v_max = k_map_semantic_int32_max;
  if (end - start == 1) {
    out[t] = cuco::pair<int64_t, int32_t>{sorted[start].first, v_max};
    return;
  }

  int32_t const anchor = sorted[start].second;
  int k = start + 1;
  while (k < end && sorted[k].second == anchor)
    ++k;
  int32_t const val = (k >= end) ? v_max : sorted[k].second;
  out[t] = cuco::pair<int64_t, int32_t>{sorted[start].first, val};
}

}  // namespace

LSM_NVMeQueueMapState *lsm_nvme_queue_map_state_new() {
  return new LSM_NVMeQueueMapState();
}

void lsm_nvme_queue_map_state_delete(LSM_NVMeQueueMapState *p) noexcept {
  delete p;
}

void lsm_nvme_queue_map_build_impl(LSM_NVMeQueueMapState *st, int cuda_device,
                                   uint64_t d_node_ids_ptr, uint64_t d_batch_idx_ptr,
                                   int32_t n, uint64_t map_capacity) {
  if (st == nullptr || n <= 0 || map_capacity == 0)
    return;
  cudaSetDevice(cuda_device);

  auto *d_ids = reinterpret_cast<int64_t *>(d_node_ids_ptr);
  auto *d_batch = reinterpret_cast<int32_t *>(d_batch_idx_ptr);

  const std::size_t cap = static_cast<std::size_t>(map_capacity);
  const std::size_t prev_cap = (st->map != nullptr) ? st->requested_capacity : 0;
  const bool legacy_no_erase =
      st->map != nullptr &&
      st->map->erased_key_sentinel() == st->map->empty_key_sentinel();
  const std::size_t new_cap = std::max(std::max(cap, prev_cap), std::size_t{1});

  if (!st->map || st->requested_capacity < cap || legacy_no_erase) {
    st->map = std::make_unique<LSM_NVMeQueueMapState::MapType>(
        cuco::extent<std::size_t>{new_cap}, cuco::empty_key<int64_t>{-1},
        cuco::empty_value<int32_t>{k_cuco_empty_payload_v},
        cuco::erased_key<int64_t>{-2});
    st->requested_capacity = new_cap;
  }

  st->map->clear();

  const std::size_t n_sz = static_cast<std::size_t>(n);
  thrust::device_vector<cuco::pair<int64_t, int32_t>> pairs(n_sz);
  auto zit = thrust::make_zip_iterator(
      thrust::make_tuple(thrust::device_pointer_cast(d_ids),
                         thrust::device_pointer_cast(d_batch)));
  thrust::transform(
      thrust::device, zit, zit + static_cast<ptrdiff_t>(n), pairs.begin(),
      [] __device__(thrust::tuple<int64_t, int32_t> const &t) {
        return cuco::pair<int64_t, int32_t>{thrust::get<0>(t), thrust::get<1>(t)};
      });

  thrust::device_vector<cuco::pair<int64_t, int32_t>> compact(n_sz);
  auto tail =
      thrust::copy_if(thrust::device, pairs.begin(), pairs.end(), compact.begin(),
                      [] __device__(cuco::pair<int64_t, int32_t> const &p) {
                        return p.first >= 0;
                      });

  const int m = static_cast<int>(thrust::distance(compact.begin(), tail));
  if (m <= 0)
    return;

  thrust::sort(thrust::device, compact.begin(), compact.begin() + m,
               [] __device__(cuco::pair<int64_t, int32_t> const &a,
                             cuco::pair<int64_t, int32_t> const &b) {
                 if (a.first != b.first)
                   return a.first < b.first;
                 return a.second < b.second;
               });

  thrust::device_vector<int> head_starts(static_cast<std::size_t>(m));
  auto head_end = thrust::copy_if(
      thrust::make_counting_iterator(0), thrust::make_counting_iterator(m),
      head_starts.begin(),
      is_segment_head{thrust::raw_pointer_cast(compact.data())});
  const int nh = static_cast<int>(thrust::distance(head_starts.begin(), head_end));
  if (nh <= 0)
    return;

  thrust::device_vector<cuco::pair<int64_t, int32_t>> uniq(
      static_cast<std::size_t>(nh));

  constexpr int threads = 256;
  int blocks = (nh + threads - 1) / threads;
  fill_node_reuse_map_entries_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(compact.data()), m,
      thrust::raw_pointer_cast(head_starts.data()), nh,
      thrust::raw_pointer_cast(uniq.data()));
  cudaDeviceSynchronize();

  st->map->insert(uniq.begin(), uniq.end());
}

void lsm_nvme_queue_map_index_map_add_impl(LSM_NVMeQueueMapState *st, int cuda_device,
                                           uint64_t d_node_ids_ptr, int32_t n,
                                           uint32_t time_step) {
  if (st == nullptr || n <= 0 || st->map == nullptr)
    return;
  cudaSetDevice(cuda_device);

  auto *d_ids = reinterpret_cast<int64_t *>(d_node_ids_ptr);
  auto ref = st->map->ref(cuco::insert_or_apply);
  int32_t const ts_i32 = static_cast<int32_t>(time_step);

  constexpr int threads = 256;
  constexpr int cg = static_cast<int>(LSM_NVMeQueueMapInsertOrApplyRef::cg_size);
  int const tiles = static_cast<int>(n);
  int const blocks = (tiles * cg + threads - 1) / threads;
  lsm_nvme_index_map_add_kernel<<<blocks, threads>>>(ref, d_ids, static_cast<int>(n), ts_i32);
  cudaDeviceSynchronize();
}

void lsm_nvme_queue_map_index_map_remove_impl(LSM_NVMeQueueMapState *st, int cuda_device,
                                              uint64_t d_node_ids_ptr, int32_t n,
                                              uint32_t time_step) {
  if (st == nullptr || n <= 0 || st->map == nullptr)
    return;
  cudaSetDevice(cuda_device);

  if (st->map->erased_key_sentinel() == st->map->empty_key_sentinel()) {
    const std::size_t c = std::max(st->requested_capacity, std::size_t{1});
    st->map = std::make_unique<LSM_NVMeQueueMapState::MapType>(
        cuco::extent<std::size_t>{c}, cuco::empty_key<int64_t>{-1},
        cuco::empty_value<int32_t>{k_cuco_empty_payload_v},
        cuco::erased_key<int64_t>{-2});
    st->requested_capacity = c;
    cudaDeviceSynchronize();
    return;
  }

  auto *d_ids = reinterpret_cast<int64_t *>(d_node_ids_ptr);
  auto ref = st->map->ref(cuco::contains, cuco::find, cuco::erase);
  int32_t const ts_i32 = static_cast<int32_t>(time_step);

  constexpr int threads = 256;
  constexpr int cg = static_cast<int>(LSM_NVMeQueueMapRemoveRef::cg_size);
  int const tiles = static_cast<int>(n);
  int const blocks = (tiles * cg + threads - 1) / threads;
  lsm_nvme_index_map_remove_kernel<<<blocks, threads>>>(ref, d_ids, static_cast<int>(n), ts_i32);
  cudaDeviceSynchronize();
}
