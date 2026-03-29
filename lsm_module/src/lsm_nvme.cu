#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include <buffer.h>
#include <cuda.h>
#include <nvm_admin.h>
#include <nvm_cmd.h>
#include <nvm_ctrl.h>
#include <nvm_error.h>
#include <nvm_io.h>
#include <nvm_parallel_queue.h>
#include <nvm_queue.h>
#include <nvm_types.h>
#include <nvm_util.h>
#include <util.h>

#include <ctrl.h>
#include <lsm_module/lsm_gnn_page_cache.h>
#include <queue.h>

#include <lsm_module/lsm_nvme.h>

namespace py = pybind11;

using TYPE = LSM_NVME_TYPE;

template <typename T = TYPE>
__global__ void read_feature_kernel(array_d_t<T> *dr, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim) {
  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if (idx_idx < num_idx) {
    bam_ptr<T> ptr(dr);

    uint64_t row_index = index_ptr[idx_idx];
    uint64_t tid = threadIdx.x % 32;

    for (; tid < static_cast<uint64_t>(dim); tid += 32) {
      T temp = ptr[(row_index)*cache_dim + tid];
      out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
    }
  }
}

template <typename T = TYPE>
__global__ void update_prefetch_timestamp_kernel(array_d_t<T> *dr,
                                                 const uint64_t *pages,
                                                 const uint32_t *timestamps,
                                                 const uint64_t *idxs, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  range_d_t<T> *r_ = dr->d_ranges;
  uint64_t page_trans = 0;
  if (!lsm_gnn::detail::logical_page_has_cache_line(r_, pages[i], page_trans))
    return;

  uint64_t update_val = (static_cast<uint64_t>(timestamps[i]) << 48) | idxs[i];
  atomicMin(reinterpret_cast<unsigned long long *>(
                &(r_->cache.cache_pages[page_trans].next_reuse)),
            static_cast<unsigned long long>(update_val));
}

template <typename T = TYPE>
__global__ void read_next_reuse_for_pages_kernel(array_d_t<T> *dr,
                                                 const int64_t *pages,
                                                 uint64_t *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  range_d_t<T> *r_ = dr->d_ranges;
  uint64_t page_trans = 0;
  uint64_t pg = static_cast<uint64_t>(pages[i]);
  if (!lsm_gnn::detail::logical_page_has_cache_line(r_, pg, page_trans)) {
    out[i] = ~0ULL;
    return;
  }

  out[i] = r_->cache.cache_pages[page_trans].next_reuse;
}

LSM_NVMeStore::~LSM_NVMeStore() {
  if (is_pvp)
    lsm_gnn::pinned_eviction_buffer_destroy(pinned_eviction_);
  for (auto *c : ctrls)
    delete c;
  ctrls.clear();
  delete h_pc;
  delete h_range;
  delete a;
  h_pc = nullptr;
  h_range = nullptr;
  a = nullptr;
}

void LSM_NVMeStore::init_controllers(int ps, uint64_t read_off,
                                     uint64_t cache_size_gb, uint64_t num_ele,
                                     uint64_t num_ssd, bool is_pvp_enable,
                                     uint32_t num_pvp_buffers_arg,
                                     uint32_t pvp_queue_depth_arg) {
  numElems = num_ele;
  n_ctrls = static_cast<uint32_t>(num_ssd);
  pageSize = static_cast<size_t>(ps);
  is_pvp = is_pvp_enable;

  const uint32_t eff_nbuf = num_pvp_buffers_arg
                                ? num_pvp_buffers_arg
                                : lsm_gnn::default_num_pvp_buffers();
  const uint32_t eff_qdepth =
      pvp_queue_depth_arg ? pvp_queue_depth_arg
                          : lsm_gnn::default_pvp_queue_depth();
  num_pvp_buffers = eff_nbuf;
  pvp_queue_depth = eff_qdepth;
  eviction_time_step = 0;
  eviction_head_ptr = 0;

  cudaSetDevice(cudaDevice);

  for (size_t i = 0; i < num_ssd; i++) {
    ctrls.push_back(new Controller(ctrls_paths[i], nvmNamespace, cudaDevice,
                                   queueDepth, numQueues));
  }

  uint64_t page_size = pageSize;
  uint64_t n_pages = cache_size_gb * 1024ULL * 1024ULL * 1024ULL / page_size;

  const uint32_t ctor_pvp_depth = is_pvp_enable ? eff_qdepth : 0u;
  h_pc = new page_cache_t(page_size, n_pages, cudaDevice, ctrls[0][0],
                          (uint64_t)64, ctrls, is_pvp_enable, eff_nbuf,
                          ctor_pvp_depth);

  uint64_t t_size = numElems * sizeof(TYPE);
  h_range = new range_t<TYPE>(
      (uint64_t)0, (uint64_t)numElems, (uint64_t)read_off,
      (uint64_t)(t_size / page_size), (uint64_t)0, (uint64_t)page_size, h_pc,
      cudaDevice, REPLICATE);

  vr.push_back(nullptr);
  vr[0] = h_range;

  a = new array_t<TYPE>(numElems, 0, vr, cudaDevice);

  if (is_pvp_enable) {
    lsm_gnn::pinned_eviction_buffer_init(pinned_eviction_, page_size, eff_nbuf,
                                         eff_qdepth);
    h_pc->bind_pinned_eviction_staging(
        pinned_eviction_.pvp_queue_counter,
        reinterpret_cast<void*>(pinned_eviction_.device_queue),
        pinned_eviction_.pvp_id_array, eff_nbuf, eff_qdepth);
    h_range->sync_embedded_page_cache_from(h_pc);
    a->resync_device_range_from(*h_range, 0);
  }
}

void LSM_NVMeStore::read_feature(uint64_t i_ptr, uint64_t i_index_ptr,
                                 int64_t num_index, int dim, int cache_dim) {
  cuda_err_chk(cudaSetDevice(static_cast<int>(cudaDevice)));
  TYPE *tensor_ptr = (TYPE *)i_ptr;
  int64_t *index_ptr = (int64_t *)i_index_ptr;

  uint64_t b_size = blkSize;
  uint64_t n_warp = b_size / 32;
  uint64_t g_size = (num_index + n_warp - 1) / n_warp;

  cuda_err_chk(cudaDeviceSynchronize());
  const bool use_pvp_path =
      (h_pc != nullptr && h_pc->is_pvp && pinned_eviction_.valid());
  if (use_pvp_path) {
    //printf("using pvp path time_step: %d, head_ptr: %d\n", eviction_time_step, eviction_head_ptr);
    lsm_gnn::read_feature_pvp_kernel<TYPE><<<g_size, b_size>>>(
        a->d_array_ptr, tensor_ptr, index_ptr, dim, num_index, cache_dim,
        eviction_time_step, eviction_head_ptr);
  } else {
    read_feature_kernel<TYPE><<<g_size, b_size>>>(
        a->d_array_ptr, tensor_ptr, index_ptr, dim, num_index, cache_dim);
  }
  cuda_err_chk(cudaDeviceSynchronize());
}

void LSM_NVMeStore::update_prefetch_timestamp(uint64_t d_pages_ptr,
                                              uint64_t d_ts_ptr,
                                              uint64_t d_idx_ptr, int n) {
  if (n <= 0 || a == nullptr)
    return;
  cuda_err_chk(cudaSetDevice(static_cast<int>(cudaDevice)));
  const uint64_t *pages = reinterpret_cast<const uint64_t *>(d_pages_ptr);
  const uint32_t *ts = reinterpret_cast<const uint32_t *>(d_ts_ptr);
  const uint64_t *idx = reinterpret_cast<const uint64_t *>(d_idx_ptr);

  constexpr int threads = 256;
  int blocks = (n + threads - 1) / threads;
  update_prefetch_timestamp_kernel<TYPE><<<blocks, threads>>>(
      a->d_array_ptr, pages, ts, idx, n);
  cuda_err_chk(cudaDeviceSynchronize());
}

void LSM_NVMeStore::read_next_reuse_for_pages(uint64_t d_pages_ptr,
                                              uint64_t d_out_ptr, int64_t n) {
  if (n <= 0 || a == nullptr)
    return;
  cuda_err_chk(cudaSetDevice(static_cast<int>(cudaDevice)));
  const int64_t *pages = reinterpret_cast<const int64_t *>(d_pages_ptr);
  uint64_t *out = reinterpret_cast<uint64_t *>(d_out_ptr);

  constexpr int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);
  read_next_reuse_for_pages_kernel<TYPE><<<blocks, threads>>>(
      a->d_array_ptr, pages, out, static_cast<int>(n));
  cuda_err_chk(cudaDeviceSynchronize());
}

void LSM_NVMeStore::PVP_prefetch(uint64_t device_ptr, uint32_t time_step) {
  if (device_ptr == 0 || !is_pvp || !pinned_eviction_.valid() ||
      num_pvp_buffers == 0 || pvp_queue_depth == 0 ||
      pinned_eviction_.host_queue == nullptr) {
    return;
  }
  cuda_err_chk(cudaSetDevice(static_cast<int>(cudaDevice)));
  const uint32_t head_ptr = time_step % num_pvp_buffers;
  const std::size_t page_sz = pageSize;
  const std::size_t bytes_per_head =
      page_sz * static_cast<std::size_t>(pvp_queue_depth);
  const std::uint8_t *src =
      reinterpret_cast<const std::uint8_t *>(pinned_eviction_.host_queue) +
      static_cast<std::size_t>(head_ptr) * bytes_per_head;
  void *dst = reinterpret_cast<void *>(device_ptr);
  cuda_err_chk(cudaMemcpy(dst, src, bytes_per_head, cudaMemcpyHostToDevice));
  cuda_err_chk(cudaDeviceSynchronize());
}

uint64_t LSM_NVMeStore::ssd_read_ops_count() const {
  if (h_pc == nullptr || h_pc->pdt.ssd_read_ops == nullptr)
    return 0;
  cuda_err_chk(cudaSetDevice(static_cast<int>(cudaDevice)));
  uint64_t v = 0;
  cuda_err_chk(cudaMemcpy(&v, h_pc->pdt.ssd_read_ops, sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));
  return v;
}

PYBIND11_MODULE(LSM_NVMe, m) {
  m.doc() = "LSM-GNN NVMe: read_feature + update_prefetch_timestamp";

  py::class_<LSM_NVMeStore, std::unique_ptr<LSM_NVMeStore, py::nodelete>>(
      m, "LSM_NVMeStore")
      .def(py::init([]() { return new LSM_NVMeStore(); }))
      .def_readwrite("cudaDevice", &LSM_NVMeStore::cudaDevice)
      .def_readonly("num_pvp_buffers", &LSM_NVMeStore::num_pvp_buffers)
      .def_readonly("pvp_queue_depth", &LSM_NVMeStore::pvp_queue_depth)
      .def_readonly("pageSize", &LSM_NVMeStore::pageSize)
      .def("init_controllers", &LSM_NVMeStore::init_controllers, py::arg("ps"),
           py::arg("read_off"), py::arg("cache_size_gb"), py::arg("num_ele"),
           py::arg("num_ssd") = 1, py::arg("is_pvp") = false,
           py::arg("num_pvp_buffers") = 0, py::arg("pvp_queue_depth") = 0)
      .def(
          "pvp_copy_device_queue_counts",
          [](const LSM_NVMeStore &s) -> py::array_t<uint32_t> {
            if (!s.is_pvp || s.pinned_eviction_.pvp_queue_counter == nullptr ||
                s.num_pvp_buffers == 0) {
              return py::array_t<uint32_t>();
            }
            py::array_t<uint32_t> out(
                {static_cast<py::ssize_t>(s.num_pvp_buffers)});
            cuda_err_chk(cudaSetDevice(s.cudaDevice));
            cuda_err_chk(cudaMemcpy(
                out.mutable_data(), s.pinned_eviction_.pvp_queue_counter,
                sizeof(uint32_t) * s.num_pvp_buffers, cudaMemcpyDeviceToHost));
            return out;
          })
      .def(
          "pvp_copy_host_meta_ids",
          [](const LSM_NVMeStore &s) -> py::array_t<uint64_t> {
            if (!s.is_pvp || s.pinned_eviction_.host_id_storage == nullptr ||
                s.num_pvp_buffers == 0 || s.pvp_queue_depth == 0) {
              return py::array_t<uint64_t>();
            }
            const size_t n =
                static_cast<size_t>(s.num_pvp_buffers) *
                static_cast<size_t>(s.pvp_queue_depth);
            py::array_t<uint64_t> out(
                {static_cast<py::ssize_t>(s.num_pvp_buffers),
                 static_cast<py::ssize_t>(s.pvp_queue_depth)});
            std::memcpy(out.mutable_data(), s.pinned_eviction_.host_id_storage,
                        n * sizeof(uint64_t));
            return out;
          })
      .def(
          "pvp_copy_host_embeddings",
          [](const LSM_NVMeStore &s) -> py::array_t<float> {
            if (!s.is_pvp || s.pinned_eviction_.host_queue == nullptr ||
                s.num_pvp_buffers == 0 || s.pvp_queue_depth == 0) {
              return py::array_t<float>();
            }
            const py::ssize_t fpe =
                static_cast<py::ssize_t>(s.pageSize / sizeof(float));
            py::array_t<float> out(
                {static_cast<py::ssize_t>(s.num_pvp_buffers),
                 static_cast<py::ssize_t>(s.pvp_queue_depth), fpe});
            const size_t nbytes =
                static_cast<size_t>(s.pageSize) *
                static_cast<size_t>(s.num_pvp_buffers) *
                static_cast<size_t>(s.pvp_queue_depth);
            std::memcpy(out.mutable_data(), s.pinned_eviction_.host_queue,
                        nbytes);
            return out;
          })
      .def_readwrite("eviction_time_step", &LSM_NVMeStore::eviction_time_step)
      .def_readwrite("eviction_head_ptr", &LSM_NVMeStore::eviction_head_ptr)
      .def("read_feature", &LSM_NVMeStore::read_feature, py::arg("tensor_ptr"),
           py::arg("index_ptr"), py::arg("num_index"), py::arg("dim"),
           py::arg("cache_dim") = 1024)
      .def("update_prefetch_timestamp", &LSM_NVMeStore::update_prefetch_timestamp,
           py::arg("d_pages_ptr"), py::arg("d_ts_ptr"), py::arg("d_idx_ptr"),
           py::arg("n"))
      .def("read_next_reuse_for_pages", &LSM_NVMeStore::read_next_reuse_for_pages,
           py::arg("d_pages_ptr"), py::arg("d_out_ptr"), py::arg("n"))
      .def("PVP_prefetch", &LSM_NVMeStore::PVP_prefetch, py::arg("device_ptr"),
           py::arg("time_step"),
           "Copy pinned PVP ring for head (time_step %% num_pvp_buffers) to device_ptr; "
           "size is pvp_queue_depth * page_size bytes.")
      .def("ssd_read_ops_count", &LSM_NVMeStore::ssd_read_ops_count,
           "Return cumulative NVMe read op count (device atomic on page cache).");
}
