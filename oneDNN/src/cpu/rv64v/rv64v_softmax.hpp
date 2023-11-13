#ifndef CPU_RV64V_RV64V_SOFTMAX_HPP
#define CPU_RV64V_RV64V_SOFTMAX_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_softmax_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct rv64_softmax_fwd_t : public primitive_t {
    struct pd_t : public cpu_softmax_fwd_pd_t {
        using cpu_softmax_fwd_pd_t::cpu_softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T("rv64:any", rv64_softmax_fwd_t);

        status_t init(engine_t *engine) {
            
            using namespace data_type;
            using skip_mask_t = primitive_attr_t::skip_mask_t;

            bool ok = is_fwd()
                    && utils::one_of(src_md()->data_type, f32, bf16, s8, u8)
                    && utils::one_of(dst_md()->data_type, f32, bf16, s8, u8)
                    && platform::has_data_type_support(src_md()->data_type)
                    && platform::has_data_type_support(dst_md()->data_type)
                    && attr()->has_default_values(skip_mask_t::oscale)
                    && attr_oscale_ok()
                    && set_default_formats() == status::success;
            if (!ok) return status::unimplemented;
            return status::success;
        }
    };

    rv64_softmax_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward_generic(ctx);
    }

private:
    status_t execute_forward_generic(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
