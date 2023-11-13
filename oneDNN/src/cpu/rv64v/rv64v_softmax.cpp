#include <cpu/rv64v/rv64v_softmax.hpp>
#include <stdio.h>
#include <stdlib.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

#define vsetvl(avl)                 __builtin_epi_vsetvl(avl, __epi_e32, __epi_m1)
#define vload(ptr)                  __builtin_epi_vload_2xf32(ptr, gvl)
#define vload_stride(ptr, stride)   __builtin_epi_vload_strided_2xf32(ptr, stride, gvl)
#define vload_indexed(ptr, a)       __builtin_epi_vload_indexed_2xf32(ptr, a, gvl)
#define vset_first(a)               __builtin_epi_vsetfirst_2xf32(a, 1)
#define vstore(ptr, v)              __builtin_epi_vstore_2xf32(ptr, v, gvl)
#define vstore_first(ptr, v)        __builtin_epi_vstore_2xf32(ptr, v, 1)
#define vistore(ptr, v)             __builtin_epi_vstore_2xi32(ptr, v, gvl)
#define vfadd(a, b)                 __builtin_epi_vfadd_2xf32(a, b, gvl)
#define vadd(a, b)                  __builtin_epi_vadd_2xi32(a, b, gvl)
#define vfsub(a, b)                 __builtin_epi_vfsub_2xf32(a, b, gvl)
#define vsub(a, b)                  __builtin_epi_vsub_2xi32(a, b, gvl)
#define vfmul(a, b)                 __builtin_epi_vfmul_2xf32(a, b, gvl)
#define vfmul_mask(a, b, c, d)      __builtin_epi_vfmul_2xf32_mask(a, b, c, d, gvl)
#define vfdiv(a, b)                 __builtin_epi_vfdiv_2xf32(a, b, gvl)
#define vfbroadcast(a)              __builtin_epi_vbroadcast_2xf32(a, gvl)
#define vbroadcast(a)               __builtin_epi_vbroadcast_2xi32(a, gvl)
#define vf2i(a)                     __builtin_epi_vfcvt_x_f_2xi32_2xf32(a, gvl)
#define vi2f(a)                     __builtin_epi_vfcvt_f_x_2xf32_2xi32(a, gvl)
#define vfmacc(a, b, c)             __builtin_epi_vfmacc_2xf32(a, b, c, gvl)
#define vmfgt(a, b)                 __builtin_epi_vmfgt_2xf32(a, b, gvl)
#define vmflt(a, b)                 __builtin_epi_vmflt_2xf32(a, b, gvl)
#define vsll(a, b)                  __builtin_epi_vsll_2xi32(a, b, gvl)
#define vsra(a, b)                  __builtin_epi_vsra_2xi32(a, b, gvl)
#define vfredmax(a, b)              __builtin_epi_vfredmax_2xf32(a, b, gvl)
#define vfredosum(a, b)             __builtin_epi_vfredosum_2xf32(a, b, gvl)
#define vfmax(a, b)                 __builtin_epi_vfmax_2xf32(a, b, gvl)

__epi_2xf32 vfexp(__epi_2xf32 va, int gvl){

    #define R_LN2f 1.442695040888963407359924681001892137426645954152985934135449406931f
    #define L2Uf 0.693145751953125f
    #define L2Lf 1.428606765330187045e-06f

    __epi_2xi32 integers;
    __epi_2xf32 q, s, u;
    
    // Decompose e^d into 2^q * e^s
    // Note: e^d = 2^(d/ln(2)) and d/ln(2) = q + s/ln(2)
    integers = vf2i(vfmul(va, vfbroadcast(R_LN2f))); // Take closest integer of d/ln(2)
    q = vi2f(integers);                              // To float    
    s = vfmacc(va, q, vfbroadcast(-L2Uf));
    s = vfmacc(s, q, vfbroadcast(-L2Lf));            // s = d - q * ln(2)

    // Apply Taylor's polynomial to get e^s
    u = vfbroadcast(0.000198527617612853646278381);
    u = vfmacc(vfbroadcast(0.00139304355252534151077271), u, s);
    u = vfmacc(vfbroadcast(0.00833336077630519866943359), u, s);
    u = vfmacc(vfbroadcast(0.0416664853692054748535156), u, s);
    u = vfmacc(vfbroadcast(0.166666671633720397949219), u, s);
    u = vfmacc(vfbroadcast(0.5), u, s);
    u = vfmacc(vfbroadcast(1), u, s);
    u = vfmacc(vfbroadcast(1), u, s);

    // Bit shift to get 2^q * e^s
    u = (__epi_2xf32)(vadd(vsll(integers, vbroadcast(23)), (__epi_2xi32)u));

    u = vfmul_mask(vfbroadcast(0), u, vfbroadcast(1), vmfgt(va, vfbroadcast(-70)));   // d < -70 => e^d = 0
    u = vfmul_mask(u, u, vfbroadcast(float('inf')), vmfgt(va, vfbroadcast(70)));      // d > 70  => e^d = inf

    return u;
}

__epi_2xf32 vflog(__epi_2xf32 va, int gvl){

    __epi_2xi32 rawIntBits, exponents;
    __epi_2xf32 mantissas, x, x2, t;

    // Decompose numbers into m * 2^e
    rawIntBits = (__epi_2xi32)va;
    exponents = vsub(vsra(rawIntBits, vbroadcast(23)), vbroadcast(0x7f));
    mantissas = (__epi_2xf32)vsub(rawIntBits, vsll(exponents, vbroadcast(23)));

    // x2 = [(m - 1)/(m + 1)]^2
    x = vfsub(mantissas, vfbroadcast(1));
    x = vfdiv(x, vfadd(mantissas, vfbroadcast(1)));
    x2 = vfmul(x, x);

    // Apply Taylor's polynomial
    t = vfbroadcast(0.2392828464508056640625f);
    t = vfmacc(vfbroadcast(0.28518211841583251953125f), t, x2);
    t = vfmacc(vfbroadcast(0.400005877017974853515625f), t, x2);
    t = vfmacc(vfbroadcast(0.666666686534881591796875f), t, x2);
    t = vfmacc(vfbroadcast(2.0f), t, x2);

    // x <- x * t + ln(2) * e
    // Note: d = m * 2^e => ln(d) = ln(m) + ln(2) * e
    x = vfadd(vfmul(x, t), vfmul(vfbroadcast(0.693147180559945286226764f), vi2f(exponents)));

    // TODO add some health conditions

    return x;
}

#define vfexp(a) vfexp(a, gvl)
#define vflog(a) vflog(a, gvl)

void rescale(const float * tensors, float * out, int size, int ntensors) {
    int i, j;
    int gvl;
    __epi_2xf32 vmax, rescaled;

    for (j = 0; j < size; j += gvl) {
        gvl = vsetvl(size - j);
        vmax = vload(tensors + j);
        for (i = 1; i < ntensors; ++i) vmax = vfmax(vload(tensors + size * i + j), vmax);
        for (i = 0; i < ntensors; ++i) {
            rescaled = vfsub(vload(tensors + size * i + j), vmax);
            vstore(out + size * i + j, rescaled);
        }
    }

    return;
}

void normalize(float * tensors, float * out, int size, int ntensors) {
    int i, j;
    int gvl;
    __epi_2xf32 sum, normalized;

    for (j = 0; j < size; j += gvl) {
        gvl = vsetvl(size - j);
        sum = vload(tensors + j);
        for (i = 1; i < ntensors; ++i) sum = vfadd(vload(tensors + size * i + j), sum);
        for (i = 0; i < ntensors; ++i) {
            normalized = vfdiv(vload(tensors + size * i + j), sum);
            vstore(out + size * i + j, normalized);
        }
    }

    return;
}

status_t
rv64_softmax_fwd_t::execute_forward_generic(const exec_ctx_t &ctx) const {
    using dtype = float;
    auto src = CTX_IN_MEM(const dtype *, DNNL_ARG_SRC); // Input tensor pointer
    auto dst = CTX_OUT_MEM(dtype *, DNNL_ARG_DST);      // Output tensor pointer

    const memory_desc_wrapper src_d(pd()->src_md());   // SRC memory descriptor
    const memory_desc_wrapper dst_d(pd()->dst_md());   // DST memory descriptor

    // Useful examples of interaction with the primitive descriptors
    const int l = src_d.nelems();               // Total size
    const dim_t MB = pd()->MB();                // Minibatch dimension size
    const dim_t C = pd()->C();                  // Channels dimension size
    const dim_t D = pd()->D();                  // Depth dimension size
    const dim_t H = pd()->H();                  // Height dimension size
    const dim_t W = pd()->W();                  // Width dimension size
    const int axis = pd()->axis();              // Softmax target axis
    const bool is_softmax = pd()->is_softmax(); // softmax or logsoftmax?

    /*
    // Useful example of interaction with memory descriptors
    int mb = 0, c = 0, d = 0, h = 0, w = 0;
    // Offset to a src tensor element at position [mb, c, d, h, w]
    size_t src_off = src_d.off(mb, c, d, h, w);
    // Compute the offset to a dst tensor element at position [mb, c, d, h, w]
    size_t dst_off = dst_d.off(mb, c, d, h, w);
    // Access both src and dst elements at the computed offset
    dst[dst_off] = src[src_off];
    */

    printf("-----\n");
    printf("Parameters:\n");
    printf("l: %d\n", l);
    printf("MB: %d\n", MB);
    printf("C: %d\n", C);
    printf("D: %d\n", D);
    printf("H: %d\n", H);
    printf("W: %d\n", W);
    printf("axis: %d\n", axis);
    printf("is_softmax: %d\n", is_softmax);
    printf("-----\n");
    
    int i, j, gvl;

    __epi_2xf32 va;

    int shape[5];
    shape[0] = MB; shape[1] = C; shape[2] = D; shape[3] = H; shape[4] = W;

    // Compute vector of indices of the first layer of the tensor
    int * indices = (int *)malloc(l * sizeof(int), 32);
    int i0, i1, i2, i3, i4;
    i = 0;
    if(axis == 0){
        for(i0 = 0; i0 < shape[0]; i0++)
        for(i1 = 0; i1 < shape[1]; i1++)
        for(i2 = 0; i2 < shape[2]; i2++)
        for(i3 = 0; i3 < shape[3]; i3++)
        for(i4 = 0; i4 < shape[4]; i4++){
            indices[i] = shape[1]*shape[2]*shape[3]*shape[4]*i0 + shape[2]*shape[3]*shape[4]*i1 + shape[3]*shape[4]*i2 + shape[4]*i3 + i4;
            ++i;
        }
    } else if(axis == 1) {
        for(i1 = 0; i1 < shape[1]; i1++)
        for(i0 = 0; i0 < shape[0]; i0++)
        for(i2 = 0; i2 < shape[2]; i2++)
        for(i3 = 0; i3 < shape[3]; i3++)
        for(i4 = 0; i4 < shape[4]; i4++){
            indices[i] = shape[1]*shape[2]*shape[3]*shape[4]*i0 + shape[2]*shape[3]*shape[4]*i1 + shape[3]*shape[4]*i2 + shape[4]*i3 + i4;
            ++i;
        }
    } else if(axis == 2) {
        for(i2 = 0; i2 < shape[2]; i2++)
        for(i0 = 0; i0 < shape[0]; i0++)
        for(i1 = 0; i1 < shape[1]; i1++)
        for(i3 = 0; i3 < shape[3]; i3++)
        for(i4 = 0; i4 < shape[4]; i4++){
            indices[i] = shape[1]*shape[2]*shape[3]*shape[4]*i0 + shape[2]*shape[3]*shape[4]*i1 + shape[3]*shape[4]*i2 + shape[4]*i3 + i4;
            ++i;
        }
    } else if(axis == 3) {
        for(i3 = 0; i3 < shape[3]; i3++)
        for(i0 = 0; i0 < shape[0]; i0++)
        for(i1 = 0; i1 < shape[1]; i1++)
        for(i2 = 0; i2 < shape[2]; i2++)
        for(i4 = 0; i4 < shape[4]; i4++){
            indices[i] = shape[1]*shape[2]*shape[3]*shape[4]*i0 + shape[2]*shape[3]*shape[4]*i1 + shape[3]*shape[4]*i2 + shape[4]*i3 + i4;
            ++i;
        }
    } else if(axis == 4) {
        for(i4 = 0; i4 < shape[4]; i4++)
        for(i0 = 0; i0 < shape[0]; i0++)
        for(i1 = 0; i1 < shape[1]; i1++)
        for(i2 = 0; i2 < shape[2]; i2++)
        for(i3 = 0; i3 < shape[3]; i3++){
            indices[i] = shape[1]*shape[2]*shape[3]*shape[4]*i0 + shape[2]*shape[3]*shape[4]*i1 + shape[3]*shape[4]*i2 + shape[4]*i3 + i4;
            ++i;
        }
    }

    // Load transposed tensor
    float * src_transposed = (float *)malloc(l * sizeof(float), 32);
    for(i=0; i < l; i+=gvl){
        gvl = vsetvl(l - i);
        va = vload_indexed(src + i, indices);
        vstore(src_transposed + i, va);
    }

    if(is_softmax){ 
        rescale(src, dst, l/shape[0], shape[0]);
        // Exponential
        for(i=0; i < l; i+=gvl){
            gvl = vsetvl(l - i);
            va = vload(dst + i);
            va = vfexp(va);
            vstore(dst + i, va);
        }
        normalize(dst, dst, l/shape[0], shape[0]);
    } else {
        float * rescaled = (float *)malloc(l * sizeof(float), 32);
        __epi_2xf32 sum;

        // Subtract maximum
        rescale(src, rescaled, l/shape[0], shape[0]);

        // Compute exponential
        for(i=0; i < l; i+=gvl){
            gvl = vsetvl(l - i);
            va = vload(rescaled + i);
            va = vfexp(va);
            vstore(dst + i, va);
        }

        for (j = 0; j < l/shape[0]; j += gvl) {
            gvl = vsetvl(l/shape[0] - j);
            sum = vload(dst + j);

            // Compute maximum following one axis
            for (i = 1; i < shape[0]; ++i) sum = vfadd(vload(dst + l/shape[0] * i + j), sum);

            // Apply log
            sum = vflog(sum);

            // rescaled_tensor - ln(sum)
            for (i = 0; i < shape[0]; ++i) {
                va = vload(rescaled + l/shape[0] * i + j);
                vstore(dst + l/shape[0] * i + j, vfsub(va, sum));
            }
        }
    }

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

/* SCALAR VERSION OF RESCALATION
int mb, c, d, h, w;
int stride[5];
stride[0] = C*D*H*W; stride[1] = D*H*W; stride[2] = H*W; stride[3] = W; stride[4] = 1;
int * shape_address[5] = {&mb, &c, &d, &h, &w};
int X[5], * x[5];

X[4] = shape[axis];
x[4] = shape_address[axis];
for(i = 0; i < 4; ++i){
    X[i] = i < axis ? shape[i] : shape[i+1];
    x[i] = i < axis ? shape_address[i] : shape_address[i+1];
}

for(*x[0] = 0; *x[0] < X[0]; ++*x[0])
for(*x[1] = 0; *x[1] < X[1]; ++*x[1])
for(*x[2] = 0; *x[2] < X[2]; ++*x[2])
for(*x[3] = 0; *x[3] < X[3]; ++*x[3]) {
    *x[4] = 0;
    max = src[C*D*H*W*mb + D*H*W*c + H*W*d + W*h + w];
    for(*x[4] = 1; *x[4] < X[4]; ++*x[4]){
        tmp = src[C*D*H*W*mb + D*H*W*c + H*W*d + W*h + w];
        if(tmp > max) max = tmp;
    }
    for(*x[4] = 0; *x[4] < X[4]; ++*x[4])
        rescaled[C*D*H*W*mb + D*H*W*c + H*W*d + W*h + w] = src[C*D*H*W*mb + D*H*W*c + H*W*d + W*h + w] - max;
}*/

/*  SCALAR VERSION OF NORMALIZATION
for(*x[0] = 0; *x[0] < X[0]; ++*x[0])
for(*x[1] = 0; *x[1] < X[1]; ++*x[1])
for(*x[2] = 0; *x[2] < X[2]; ++*x[2])
for(*x[3] = 0; *x[3] < X[3]; ++*x[3]) {
    tmp = 0;
    for(*x[4] = 0; *x[4] < X[4]; ++*x[4]) tmp += exponential[C*D*H*W*mb + D*H*W*c + H*W*d + W*h + w];
    for(*x[4] = 0; *x[4] < X[4]; ++*x[4]) exponential[C*D*H*W*mb + D*H*W*c + H*W*d + W*h + w] /= tmp;
}*/

/*
// Compute vmax
vmax = vset_first(src[0]);
for (i = 0; i < l; i += gvl) {
    gvl = vsetvl(l - i);
    va = vload(src + i);
    vmax = vfredmax(va, vmax);
}
vstore_first(&max, vmax);
*/

/*
printf("RESCALED:\n");
for(i=0; i<MB; i++){
    for(j=0; j<C; j++) printf("%10f ", rescaled[C*i + j]);
    printf("\n");
}
*/
