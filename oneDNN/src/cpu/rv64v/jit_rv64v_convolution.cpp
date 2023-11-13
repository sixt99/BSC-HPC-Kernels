#include "cpu/rv64v/jit_rv64v_convolution.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

template <data_type_t T>
status_t
jit_rv64v_convolution_fwd_t<T>::do_execute(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t* const, DNNL_ARG_SRC);
    auto wei = CTX_IN_MEM(const data_t* const, DNNL_ARG_WEIGHTS);
    auto dst = CTX_OUT_MEM(data_t* const, DNNL_ARG_DST);
    auto const src_mb_size = pd()->IC() * pd()->IH() * pd()->IW();
    auto const dst_mb_size = pd()->OC() * pd()->OH() * pd()->OW();

    // #pragma omp parallel for schedule(static)
    for (int n = 0; n < pd()->MB(); ++n) {
        auto const pdst = &dst[n*dst_mb_size];
        auto const psrc = &src[n*src_mb_size];
        for(size_t i = 0; i < schedule.N; ++i)
            call_schedule(schedule, i, n, pdst, psrc, wei);
    }
    return status::success;
}

template <data_type_t T>
status_t
jit_rv64v_convolution_bwd_data_t<T>::do_execute(const exec_ctx_t &ctx) const {
    auto src = CTX_OUT_MEM(data_t * const, DNNL_ARG_DIFF_SRC);
    auto wei = CTX_IN_MEM(const data_t * const, DNNL_ARG_WEIGHTS);
    auto dst = CTX_IN_MEM(const data_t * const, DNNL_ARG_DIFF_DST);
    auto const src_mb_size = pd()->IC() * pd()->IH() * pd()->IW();
    auto const dst_mb_size = pd()->OC() * pd()->OH() * pd()->OW();
    // #pragma omp parallel for schedule(static)
    for (int n = 0; n < pd()->MB(); ++n) {
        auto const pdst = &dst[n*dst_mb_size];
        auto const psrc = &src[n*src_mb_size];
        for(size_t i = 0; i < schedule.N; ++i)
            call_schedule(schedule, i, n, pdst, psrc, wei);
    }
    return status::success;
}

template <data_type_t T>
status_t
jit_rv64v_convolution_bwd_weights_t<T>::do_execute(const exec_ctx_t &ctx) const {
    auto wei = CTX_IN_MEM(data_t*, DNNL_ARG_DIFF_WEIGHTS);
    auto src = CTX_IN_MEM(const data_t* const, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(const data_t* const, DNNL_ARG_DIFF_DST);
    auto const src_mb_size = pd()->IC() * pd()->IH() * pd()->IW();
    auto const dst_mb_size = pd()->OC() * pd()->OH() * pd()->OW();
    // #pragma omp parallel for schedule(static)
    for (int n = 0; n < pd()->MB(); ++n) {
        auto const pdst = &dst[n*dst_mb_size];
        auto const psrc = &src[n*src_mb_size];
        for(size_t i = 0; i < schedule.N; ++i)
            call_schedule(schedule, i, n, pdst, psrc, wei);
    }
    return status::success;
}

template struct jit_rv64v_convolution_fwd_t<data_type::f32>;
template struct jit_rv64v_convolution_bwd_data_t<data_type::f32>;
template struct jit_rv64v_convolution_bwd_weights_t<data_type::f32>;

} // rv64
} // cpu
} // impl
} // dnnl
