#ifndef CPU_RV64V_JIT_CONVOLUTION_KERNEL_FWDD_HPP
#define CPU_RV64V_JIT_CONVOLUTION_KERNEL_FWDD_HPP

#include "cpu/rv64v/jit/jit_assembler.hpp"
#include "cpu/rv64v/jit/convolution/driver.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

/// A structure to identify a micro-kernel solving a sub-convolution
/// Rationale: The convolution micro-kernel computes a convolution over a
/// sub-tensor with shape dictated by the register block optimization across
/// the spatial domain.
/// For a given convolution problem, the sub-tensor shape may not segment the
/// spatial domain in equal parts.
/// The JIT driver must create one JIT functions for each sub-tensor shape
/// (register block optimization) to solve the convolution problem.
/// The following structure serve to uniquely identify a micro-kernel from a
/// set of micro-kernels generated with the same set of configurations,
/// enabling to associate it with a sub-tensor.
/// @details
/// The defining characteristic is the effective register block shape for the
/// micro-kernel, and wheter or not this shape overlaps the tensor padding,
/// as the latter condition might cause some computations to be skipped.
struct kernel_traits_t {
    int erbw; // Effective register block widht
    int erbc; // Effective register block channel size
    int rbpadT; // H axis padding overlap to the top of the register block
    int rbpadB; // H axis padding overlap to the bot of the register block
    int rbpadR; // W axis padding overlap to the right of the register block
    int rbpadL; // W axis padding overlap to the left of the register block

    bool operator ==(const kernel_traits_t &o) {
        return erbw == o.erbw && erbc == o.erbc
            && rbpadT == o.rbpadT && rbpadB == o.rbpadB
            && rbpadR == o.rbpadR && rbpadL == o.rbpadL;
    }
};

struct jit_convolution_kernel_t : public jit_assembler {
private:
    jit_convolution_configuration_t cfg;
    kernel_traits_t traits;

public:
    jit_convolution_kernel_t(const jit_convolution_configuration_t &c,
                                  const kernel_traits_t &t)
        : jit_assembler(), cfg(c), traits(t) {}
    void code();

private:
    const int imm_range = imm12_max() - imm_min();

    void fwdd_inner_loops(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp);
    void bwdd_inner_loops(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp);
    void bwdw_inner_loops(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp);
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64V_JIT_JIT_CONVOLUTION_KERNEL_FWDD_HPP