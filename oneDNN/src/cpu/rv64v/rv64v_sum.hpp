#ifndef CPU_RV64V_RV64V_SUM_HPP
#define CPU_RV64V_RV64V_SUM_HPP

#include "common/engine.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/reorder.hpp"
#include "common/reorder_pd.hpp"

#include "cpu/cpu_sum_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct rv64_sum_t : public primitive_t {
    struct pd_t : public cpu_sum_pd_t {
        using cpu_sum_pd_t::cpu_sum_pd_t;

        pd_t(const pd_t &rhs) = default;

        DECLARE_SUM_PD_T("rv64v:any", rv64_sum_t);

        status_t init(engine_t *engine) {

            bool ok = cpu_sum_pd_t::init(engine) == status::success;
            if (!ok) return status::unimplemented;

            if (has_zero_dim_memory()) return status::success;

            reorder_pds_.resize(n_ + need_output_reorder());
            for (int i = 0; i < n_; ++i) {

                primitive_attr_t r_attr;
                r_attr.output_scales_.set(scales_[i]);
                if (i != 0) r_attr.post_ops_.append_sum(1.0);
                CHECK(reorder_primitive_desc_create(reorder_pds_[i], engine,
                        src_md(i), dst_acc_md(), &r_attr));
            }

            if (need_output_reorder()) {
                CHECK(reorder_primitive_desc_create(
                        reorder_pds_[n_], engine, dst_acc_md(), dst_md()));
            }

            init_scratchpad();
            return status::success;
        }

        std::vector<std::shared_ptr<primitive_desc_t>> reorder_pds_;

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            if (need_output_reorder()) {
                const memory_desc_wrapper dst_acc_d(dst_acc_md());
                scratchpad.book(key_sum_reduction, dst_acc_d.size(), 1,
                        dst_acc_d.data_type_size());
            }

            for (size_t i = 0; i < reorder_pds_.size(); i++) {
                scratchpad.book(key_nested_multiple + (int)i,
                        reorder_pds_[i]->scratchpad_registry());
            }
        };
    };

    rv64_sum_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        const size_t n = pd()->reorder_pds_.size();
        reorders_.resize(n);
        for (size_t i = 0; i < n; ++i)
            pd()->reorder_pds_[i]->create_primitive(reorders_[i], engine);
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward_generic(ctx);
    }


private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_forward_generic(const exec_ctx_t &ctx) const;
    std::vector<std::shared_ptr<primitive_t>> reorders_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif