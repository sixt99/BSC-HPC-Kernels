#include <cpu/rv64v/rv64v_sum.hpp>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

status_t rv64_sum_t::execute_forward_generic(const exec_ctx_t &ctx) const {
    using dtype = float;
    using vf32 = __epi_2xf32;
    #define vsetvl(avl)    __builtin_epi_vsetvl(avl, __epi_e32, __epi_m1)
    #define vload(ptr)     __builtin_epi_vload_2xf32(ptr, gvl)
    #define vstore(ptr, v) __builtin_epi_vstore_2xf32(ptr, v, gvl)
    #define vfadd(a, b)    __builtin_epi_vfadd_2xf32(a, b, gvl)

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const int ntensors = pd()->n_inputs();
    const int size = src_d.nelems();
    
    auto dst = CTX_OUT_MEM(dtype *, DNNL_ARG_DST);
    int gvl;

    // Iterate over the gvl-sized batches
    for (int j = 0; j < size; j += gvl) {
        vf32 va, sum;
        
        auto src = CTX_IN_MEM(const dtype *, DNNL_ARG_MULTIPLE_SRC);

        gvl = vsetvl(size - j);
        sum = vload(src + j); // Initialize to the first tensor

        // Iterate over the number of tensors
        for (int i = 1; i < ntensors; i++) {
            src = CTX_IN_MEM(const dtype *, DNNL_ARG_MULTIPLE_SRC + i);
            va = vload(src + j);
            sum = vfadd(va, sum);
        }

        // After the sum has been completed, store in out
        vstore(dst + j, sum);
    }

    #undef vsetvl
    #undef vload
    #undef vstore
    #undef vfadd

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl