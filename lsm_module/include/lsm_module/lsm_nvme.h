#ifndef LSM_MODULE_LSM_NVME_H
#define LSM_MODULE_LSM_NVME_H

#include <cstdint>
#include <vector>

#include <lsm_module/lsm_gnn_page_cache.h>

#define LSM_NVME_TYPE float

struct LSM_NVMeStore {
  const char *const ctrls_paths[5] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2",
                                      "/dev/libnvm3", "/dev/libnvm4"};

  uint32_t cudaDevice = 0;
  uint32_t nvmNamespace = 1;
  uint32_t n_ctrls = 1;

  size_t blkSize = 128;
  size_t numThreads = 64;
  size_t queueDepth = 1024;
  size_t numQueues = 128;
  size_t pageSize = 4096;
  uint64_t numElems = 0;

  std::vector<Controller *> ctrls;
  page_cache_t *h_pc = nullptr;
  range_t<LSM_NVME_TYPE> *h_range = nullptr;
  std::vector<range_t<LSM_NVME_TYPE> *> vr;
  array_t<LSM_NVME_TYPE> *a = nullptr;

  /**
   * PVP (prefetch-aware eviction): page_cache_t is built with is_pvp; host allocates
   * pinned_eviction_buffer; read_feature uses read_feature_pvp_kernel when is_pvp.
   */
  bool is_pvp = false;
  lsm_gnn::pinned_eviction_buffer<LSM_NVME_TYPE> pinned_eviction_;
  /** Effective geometry after init (0 args → lsm_gnn defaults). */
  uint32_t num_pvp_buffers = 0;
  /** Same depth for pinned eviction ring and device PVP meta/loc queues. */
  uint32_t pvp_queue_depth = 0;
  uint8_t eviction_time_step = 0;
  uint32_t eviction_head_ptr = 0;

  LSM_NVMeStore() = default;

  ~LSM_NVMeStore();

  /** Page-cache line count: ``cache_size_gb * 2^30 / page_size``. */
  void init_controllers(int ps, uint64_t read_off, uint64_t cache_size_gb,
                        uint64_t num_ele, uint64_t num_ssd = 1, bool is_pvp_enable = false,
                        uint32_t num_pvp_buffers_arg = 0, uint32_t pvp_queue_depth_arg = 0);

  /** Uses PVP kernel when h_pc->is_pvp and pinned buffer valid; else bam_ptr path. */
  void read_feature(uint64_t tensor_ptr, uint64_t index_ptr, int64_t num_index,
                    int dim, int cache_dim);

  void update_prefetch_timestamp(uint64_t d_pages_ptr, uint64_t d_ts_ptr,
                                 uint64_t d_idx_ptr, int n);

  void read_next_reuse_for_pages(uint64_t d_pages_ptr, uint64_t d_out_ptr,
                                 int64_t n);

  /**
   * Copy one PVP ring slice (``pvp_queue_depth`` pages for head ``time_step % num_pvp_buffers``)
   * from pinned host staging to ``device_ptr`` (CUDA device pointer as ``uint64_t``).
   */
  void PVP_prefetch(uint64_t device_ptr, uint32_t time_step);

  /** Host copy of device ``page_cache_d_t::ssd_read_ops`` (NVMe read command count). */
  uint64_t ssd_read_ops_count() const;
};

#endif /* LSM_MODULE_LSM_NVME_H */
