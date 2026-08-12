/*
 * Copyright © 2026, VideoLAN and dav1d authors
 * Copyright © 2026, Zhou Qiankang <wszqkzqk@qq.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Inverse transforms (src/itx_tmpl.c driver + src/itx_1d.c 1D kernels)
// implemented with Google Highway: one source is compiled per SIMD target and
// the best one supported by the CPU is selected at runtime
// (HWY_DYNAMIC_DISPATCH). Bit-exact with the C code: every arithmetic
// expression of the scalar 1D kernels is reproduced with wrapping int32 lane
// ops, including the (c - 4096) factoring that keeps the C code away from
// signed-overflow UB (see the comment in src/itx_1d.c), so identical inputs
// give identical outputs in every lane.

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/hwy/itx.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "src/hwy/common.h"

// src/scan.c. Populated by dav1d_init_last_nonzero_col_from_eob_tables(), which
// dav1d_itx_dsp_init() runs whenever any function installed here can survive
// (if arch asm covered every size it would also have replaced these pointers).
extern "C" const uint8_t *const dav1d_last_nonzero_col_from_eob[19];

HWY_BEFORE_NAMESPACE();

namespace dav1d {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// (a * ca + b * cb + RND) >> SH and (a * ca + RND) >> SH with wrapping int32
// lanes; wrapping addition is associative, so operand order is irrelevant.
template <int SH, int RND, class D>
static inline hn::VFromD<D> dot2(const D d, const hn::VFromD<D> a, const int ca,
                                 const hn::VFromD<D> b, const int cb)
{
    return hn::ShiftRight<SH>(hn::Add(hn::Add(hn::Mul(a, hn::Set(d, ca)),
                                              hn::Mul(b, hn::Set(d, cb))),
                                      hn::Set(d, RND)));
}

template <int SH, int RND, class D>
static inline hn::VFromD<D> dot1(const D d, const hn::VFromD<D> a, const int ca)
{
    return hn::ShiftRight<SH>(hn::Add(hn::Mul(a, hn::Set(d, ca)),
                                      hn::Set(d, RND)));
}

// (v * 181 + 128) >> 8
template <class D>
static inline hn::VFromD<D> rot181(const D d, const hn::VFromD<D> v)
{
    return dot1<8, 128>(d, v, 181);
}

template <class V>
static inline V cadd(const V a, const V b, const V lo, const V hi)
{
    return hn::Clamp(hn::Add(a, b), lo, hi);
}

template <class V>
static inline V csub(const V a, const V b, const V lo, const V hi)
{
    return hn::Clamp(hn::Sub(a, b), lo, hi);
}

// Port of inv_dct4_1d_internal_c (src/itx_1d.c). One transform per lane; c[k]
// below is the vector of element k of every transform in the batch.
template <class D, bool TX64>
static inline void hwy_dct4(const D d, hn::VFromD<D> *const c, const int stride,
                            const hn::VFromD<D> lo, const hn::VFromD<D> hi)
{
    using V = hn::VFromD<D>;
    const V in0 = c[0 * stride], in1 = c[1 * stride];

    V t0, t1, t2, t3;
    if constexpr (TX64) {
        t0 = t1 = rot181(d, in0);
        t2 = dot1<12, 2048>(d, in1, 1567);
        t3 = dot1<12, 2048>(d, in1, 3784);
    } else {
        const V in2 = c[2 * stride], in3 = c[3 * stride];

        t0 = rot181(d, hn::Add(in0, in2));
        t1 = rot181(d, hn::Sub(in0, in2));
        t2 = hn::Sub(dot2<12, 2048>(d, in1, 1567, in3, -(3784 - 4096)), in3);
        t3 = hn::Add(dot2<12, 2048>(d, in1, 3784 - 4096, in3, 1567), in1);
    }

    c[0 * stride] = cadd(t0, t3, lo, hi);
    c[1 * stride] = cadd(t1, t2, lo, hi);
    c[2 * stride] = csub(t1, t2, lo, hi);
    c[3 * stride] = csub(t0, t3, lo, hi);
}

// Port of inv_dct8_1d_internal_c.
template <class D, bool TX64>
static inline void hwy_dct8(const D d, hn::VFromD<D> *const c, const int stride,
                            const hn::VFromD<D> lo, const hn::VFromD<D> hi)
{
    using V = hn::VFromD<D>;
    hwy_dct4<D, TX64>(d, c, stride << 1, lo, hi);

    const V in1 = c[1 * stride], in3 = c[3 * stride];

    V t4a, t5a, t6a, t7a;
    if constexpr (TX64) {
        t4a = dot1<12, 2048>(d, in1, 799);
        t5a = dot1<12, 2048>(d, in3, -2276);
        t6a = dot1<12, 2048>(d, in3, 3406);
        t7a = dot1<12, 2048>(d, in1, 4017);
    } else {
        const V in5 = c[5 * stride], in7 = c[7 * stride];

        t4a = hn::Sub(dot2<12, 2048>(d, in1, 799, in7, -(4017 - 4096)), in7);
        t5a = dot2<11, 1024>(d, in5, 1703, in3, -1138);
        t6a = dot2<11, 1024>(d, in5, 1138, in3, 1703);
        t7a = hn::Add(dot2<12, 2048>(d, in1, 4017 - 4096, in7, 799), in1);
    }

    const V t4 = cadd(t4a, t5a, lo, hi);
    t5a = csub(t4a, t5a, lo, hi);
    const V t7 = cadd(t7a, t6a, lo, hi);
    t6a = csub(t7a, t6a, lo, hi);

    const V t5 = rot181(d, hn::Sub(t6a, t5a));
    const V t6 = rot181(d, hn::Add(t6a, t5a));

    const V t0 = c[0 * stride];
    const V t1 = c[2 * stride];
    const V t2 = c[4 * stride];
    const V t3 = c[6 * stride];

    c[0 * stride] = cadd(t0, t7, lo, hi);
    c[1 * stride] = cadd(t1, t6, lo, hi);
    c[2 * stride] = cadd(t2, t5, lo, hi);
    c[3 * stride] = cadd(t3, t4, lo, hi);
    c[4 * stride] = csub(t3, t4, lo, hi);
    c[5 * stride] = csub(t2, t5, lo, hi);
    c[6 * stride] = csub(t1, t6, lo, hi);
    c[7 * stride] = csub(t0, t7, lo, hi);
}

// Port of inv_dct16_1d_internal_c.
template <class D, bool TX64>
static inline void hwy_dct16(const D d, hn::VFromD<D> *const c, const int stride,
                             const hn::VFromD<D> lo, const hn::VFromD<D> hi)
{
    using V = hn::VFromD<D>;
    hwy_dct8<D, TX64>(d, c, stride << 1, lo, hi);

    const V in1 = c[1 * stride], in3 = c[3 * stride];
    const V in5 = c[5 * stride], in7 = c[7 * stride];

    V t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a;
    if constexpr (TX64) {
        t8a  = dot1<12, 2048>(d, in1, 401);
        t9a  = dot1<12, 2048>(d, in7, -2598);
        t10a = dot1<12, 2048>(d, in5, 1931);
        t11a = dot1<12, 2048>(d, in3, -1189);
        t12a = dot1<12, 2048>(d, in3, 3920);
        t13a = dot1<12, 2048>(d, in5, 3612);
        t14a = dot1<12, 2048>(d, in7, 3166);
        t15a = dot1<12, 2048>(d, in1, 4076);
    } else {
        const V in9  = c[ 9 * stride], in11 = c[11 * stride];
        const V in13 = c[13 * stride], in15 = c[15 * stride];

        t8a  = hn::Sub(dot2<12, 2048>(d, in1, 401, in15, -(4076 - 4096)), in15);
        t9a  = dot2<11, 1024>(d, in9, 1583, in7, -1299);
        t10a = hn::Sub(dot2<12, 2048>(d, in5, 1931, in11, -(3612 - 4096)), in11);
        t11a = hn::Add(dot2<12, 2048>(d, in13, 3920 - 4096, in3, -1189), in13);
        t12a = hn::Add(dot2<12, 2048>(d, in13, 1189, in3, 3920 - 4096), in3);
        t13a = hn::Add(dot2<12, 2048>(d, in5, 3612 - 4096, in11, 1931), in5);
        t14a = dot2<11, 1024>(d, in9, 1299, in7, 1583);
        t15a = hn::Add(dot2<12, 2048>(d, in1, 4076 - 4096, in15, 401), in1);
    }

    V t8  = cadd(t8a, t9a, lo, hi);
    V t9  = csub(t8a, t9a, lo, hi);
    V t10 = csub(t11a, t10a, lo, hi);
    V t11 = cadd(t11a, t10a, lo, hi);
    V t12 = cadd(t12a, t13a, lo, hi);
    V t13 = csub(t12a, t13a, lo, hi);
    V t14 = csub(t15a, t14a, lo, hi);
    V t15 = cadd(t15a, t14a, lo, hi);

    t9a  = hn::Sub(dot2<12, 2048>(d, t14, 1567, t9, -(3784 - 4096)), t9);
    t14a = hn::Add(dot2<12, 2048>(d, t14, 3784 - 4096, t9, 1567), t14);
    t10a = hn::Sub(dot2<12, 2048>(d, t13, -(3784 - 4096), t10, -1567), t13);
    t13a = hn::Sub(dot2<12, 2048>(d, t13, 1567, t10, -(3784 - 4096)), t10);

    t8a  = cadd(t8, t11, lo, hi);
    t9   = cadd(t9a, t10a, lo, hi);
    t10  = csub(t9a, t10a, lo, hi);
    t11a = csub(t8, t11, lo, hi);
    t12a = csub(t15, t12, lo, hi);
    t13  = csub(t14a, t13a, lo, hi);
    t14  = cadd(t14a, t13a, lo, hi);
    t15a = cadd(t15, t12, lo, hi);

    t10a = rot181(d, hn::Sub(t13, t10));
    t13a = rot181(d, hn::Add(t13, t10));
    t11  = rot181(d, hn::Sub(t12a, t11a));
    t12  = rot181(d, hn::Add(t12a, t11a));

    const V t0 = c[ 0 * stride];
    const V t1 = c[ 2 * stride];
    const V t2 = c[ 4 * stride];
    const V t3 = c[ 6 * stride];
    const V t4 = c[ 8 * stride];
    const V t5 = c[10 * stride];
    const V t6 = c[12 * stride];
    const V t7 = c[14 * stride];

    c[ 0 * stride] = cadd(t0, t15a, lo, hi);
    c[ 1 * stride] = cadd(t1, t14, lo, hi);
    c[ 2 * stride] = cadd(t2, t13a, lo, hi);
    c[ 3 * stride] = cadd(t3, t12, lo, hi);
    c[ 4 * stride] = cadd(t4, t11, lo, hi);
    c[ 5 * stride] = cadd(t5, t10a, lo, hi);
    c[ 6 * stride] = cadd(t6, t9, lo, hi);
    c[ 7 * stride] = cadd(t7, t8a, lo, hi);
    c[ 8 * stride] = csub(t7, t8a, lo, hi);
    c[ 9 * stride] = csub(t6, t9, lo, hi);
    c[10 * stride] = csub(t5, t10a, lo, hi);
    c[11 * stride] = csub(t4, t11, lo, hi);
    c[12 * stride] = csub(t3, t12, lo, hi);
    c[13 * stride] = csub(t2, t13a, lo, hi);
    c[14 * stride] = csub(t1, t14, lo, hi);
    c[15 * stride] = csub(t0, t15a, lo, hi);
}

// Port of inv_dct32_1d_internal_c.
template <class D, bool TX64>
static inline void hwy_dct32(const D d, hn::VFromD<D> *const c, const int stride,
                             const hn::VFromD<D> lo, const hn::VFromD<D> hi)
{
    using V = hn::VFromD<D>;
    hwy_dct16<D, TX64>(d, c, stride << 1, lo, hi);

    const V in1  = c[ 1 * stride], in3  = c[ 3 * stride];
    const V in5  = c[ 5 * stride], in7  = c[ 7 * stride];
    const V in9  = c[ 9 * stride], in11 = c[11 * stride];
    const V in13 = c[13 * stride], in15 = c[15 * stride];

    V t16a, t17a, t18a, t19a, t20a, t21a, t22a, t23a;
    V t24a, t25a, t26a, t27a, t28a, t29a, t30a, t31a;
    if constexpr (TX64) {
        t16a = dot1<12, 2048>(d, in1, 201);
        t17a = dot1<12, 2048>(d, in15, -2751);
        t18a = dot1<12, 2048>(d, in9, 1751);
        t19a = dot1<12, 2048>(d, in7, -1380);
        t20a = dot1<12, 2048>(d, in5, 995);
        t21a = dot1<12, 2048>(d, in11, -2106);
        t22a = dot1<12, 2048>(d, in13, 2440);
        t23a = dot1<12, 2048>(d, in3, -601);
        t24a = dot1<12, 2048>(d, in3, 4052);
        t25a = dot1<12, 2048>(d, in13, 3290);
        t26a = dot1<12, 2048>(d, in11, 3513);
        t27a = dot1<12, 2048>(d, in5, 3973);
        t28a = dot1<12, 2048>(d, in7, 3857);
        t29a = dot1<12, 2048>(d, in9, 3703);
        t30a = dot1<12, 2048>(d, in15, 3035);
        t31a = dot1<12, 2048>(d, in1, 4091);
    } else {
        const V in17 = c[17 * stride], in19 = c[19 * stride];
        const V in21 = c[21 * stride], in23 = c[23 * stride];
        const V in25 = c[25 * stride], in27 = c[27 * stride];
        const V in29 = c[29 * stride], in31 = c[31 * stride];

        t16a = hn::Sub(dot2<12, 2048>(d, in1, 201, in31, -(4091 - 4096)), in31);
        t17a = hn::Add(dot2<12, 2048>(d, in17, 3035 - 4096, in15, -2751), in17);
        t18a = hn::Sub(dot2<12, 2048>(d, in9, 1751, in23, -(3703 - 4096)), in23);
        t19a = hn::Add(dot2<12, 2048>(d, in25, 3857 - 4096, in7, -1380), in25);
        t20a = hn::Sub(dot2<12, 2048>(d, in5, 995, in27, -(3973 - 4096)), in27);
        t21a = hn::Add(dot2<12, 2048>(d, in21, 3513 - 4096, in11, -2106), in21);
        t22a = dot2<11, 1024>(d, in13, 1220, in19, -1645);
        t23a = hn::Add(dot2<12, 2048>(d, in29, 4052 - 4096, in3, -601), in29);
        t24a = hn::Add(dot2<12, 2048>(d, in29, 601, in3, 4052 - 4096), in3);
        t25a = dot2<11, 1024>(d, in13, 1645, in19, 1220);
        t26a = hn::Add(dot2<12, 2048>(d, in21, 2106, in11, 3513 - 4096), in11);
        t27a = hn::Add(dot2<12, 2048>(d, in5, 3973 - 4096, in27, 995), in5);
        t28a = hn::Add(dot2<12, 2048>(d, in25, 1380, in7, 3857 - 4096), in7);
        t29a = hn::Add(dot2<12, 2048>(d, in9, 3703 - 4096, in23, 1751), in9);
        t30a = hn::Add(dot2<12, 2048>(d, in17, 2751, in15, 3035 - 4096), in15);
        t31a = hn::Add(dot2<12, 2048>(d, in1, 4091 - 4096, in31, 201), in1);
    }

    V t16 = cadd(t16a, t17a, lo, hi);
    V t17 = csub(t16a, t17a, lo, hi);
    V t18 = csub(t19a, t18a, lo, hi);
    V t19 = cadd(t19a, t18a, lo, hi);
    V t20 = cadd(t20a, t21a, lo, hi);
    V t21 = csub(t20a, t21a, lo, hi);
    V t22 = csub(t23a, t22a, lo, hi);
    V t23 = cadd(t23a, t22a, lo, hi);
    V t24 = cadd(t24a, t25a, lo, hi);
    V t25 = csub(t24a, t25a, lo, hi);
    V t26 = csub(t27a, t26a, lo, hi);
    V t27 = cadd(t27a, t26a, lo, hi);
    V t28 = cadd(t28a, t29a, lo, hi);
    V t29 = csub(t28a, t29a, lo, hi);
    V t30 = csub(t31a, t30a, lo, hi);
    V t31 = cadd(t31a, t30a, lo, hi);

    t17a = hn::Sub(dot2<12, 2048>(d, t30, 799, t17, -(4017 - 4096)), t17);
    t30a = hn::Add(dot2<12, 2048>(d, t30, 4017 - 4096, t17, 799), t30);
    t18a = hn::Sub(dot2<12, 2048>(d, t29, -(4017 - 4096), t18, -799), t29);
    t29a = hn::Sub(dot2<12, 2048>(d, t29, 799, t18, -(4017 - 4096)), t18);
    t21a = dot2<11, 1024>(d, t26, 1703, t21, -1138);
    t26a = dot2<11, 1024>(d, t26, 1138, t21, 1703);
    t22a = dot2<11, 1024>(d, t25, -1138, t22, -1703);
    t25a = dot2<11, 1024>(d, t25, 1703, t22, -1138);

    t16a = cadd(t16, t19, lo, hi);
    t17  = cadd(t17a, t18a, lo, hi);
    t18  = csub(t17a, t18a, lo, hi);
    t19a = csub(t16, t19, lo, hi);
    t20a = csub(t23, t20, lo, hi);
    t21  = csub(t22a, t21a, lo, hi);
    t22  = cadd(t22a, t21a, lo, hi);
    t23a = cadd(t23, t20, lo, hi);
    t24a = cadd(t24, t27, lo, hi);
    t25  = cadd(t25a, t26a, lo, hi);
    t26  = csub(t25a, t26a, lo, hi);
    t27a = csub(t24, t27, lo, hi);
    t28a = csub(t31, t28, lo, hi);
    t29  = csub(t30a, t29a, lo, hi);
    t30  = cadd(t30a, t29a, lo, hi);
    t31a = cadd(t31, t28, lo, hi);

    t18a = hn::Sub(dot2<12, 2048>(d, t29, 1567, t18, -(3784 - 4096)), t18);
    t29a = hn::Add(dot2<12, 2048>(d, t29, 3784 - 4096, t18, 1567), t29);
    t19  = hn::Sub(dot2<12, 2048>(d, t28a, 1567, t19a, -(3784 - 4096)), t19a);
    t28  = hn::Add(dot2<12, 2048>(d, t28a, 3784 - 4096, t19a, 1567), t28a);
    t20  = hn::Sub(dot2<12, 2048>(d, t27a, -(3784 - 4096), t20a, -1567), t27a);
    t27  = hn::Sub(dot2<12, 2048>(d, t27a, 1567, t20a, -(3784 - 4096)), t20a);
    t21a = hn::Sub(dot2<12, 2048>(d, t26, -(3784 - 4096), t21, -1567), t26);
    t26a = hn::Sub(dot2<12, 2048>(d, t26, 1567, t21, -(3784 - 4096)), t21);

    t16  = cadd(t16a, t23a, lo, hi);
    t17a = cadd(t17, t22, lo, hi);
    t18  = cadd(t18a, t21a, lo, hi);
    t19a = cadd(t19, t20, lo, hi);
    t20a = csub(t19, t20, lo, hi);
    t21  = csub(t18a, t21a, lo, hi);
    t22a = csub(t17, t22, lo, hi);
    t23  = csub(t16a, t23a, lo, hi);
    t24  = csub(t31a, t24a, lo, hi);
    t25a = csub(t30, t25, lo, hi);
    t26  = csub(t29a, t26a, lo, hi);
    t27a = csub(t28, t27, lo, hi);
    t28a = cadd(t28, t27, lo, hi);
    t29  = cadd(t29a, t26a, lo, hi);
    t30a = cadd(t30, t25, lo, hi);
    t31  = cadd(t31a, t24a, lo, hi);

    t20  = rot181(d, hn::Sub(t27a, t20a));
    t27  = rot181(d, hn::Add(t27a, t20a));
    t21a = rot181(d, hn::Sub(t26, t21));
    t26a = rot181(d, hn::Add(t26, t21));
    t22  = rot181(d, hn::Sub(t25a, t22a));
    t25  = rot181(d, hn::Add(t25a, t22a));
    t23a = rot181(d, hn::Sub(t24, t23));
    t24a = rot181(d, hn::Add(t24, t23));

    const V t0  = c[ 0 * stride];
    const V t1  = c[ 2 * stride];
    const V t2  = c[ 4 * stride];
    const V t3  = c[ 6 * stride];
    const V t4  = c[ 8 * stride];
    const V t5  = c[10 * stride];
    const V t6  = c[12 * stride];
    const V t7  = c[14 * stride];
    const V t8  = c[16 * stride];
    const V t9  = c[18 * stride];
    const V t10 = c[20 * stride];
    const V t11 = c[22 * stride];
    const V t12 = c[24 * stride];
    const V t13 = c[26 * stride];
    const V t14 = c[28 * stride];
    const V t15 = c[30 * stride];

    c[ 0 * stride] = cadd(t0, t31, lo, hi);
    c[ 1 * stride] = cadd(t1, t30a, lo, hi);
    c[ 2 * stride] = cadd(t2, t29, lo, hi);
    c[ 3 * stride] = cadd(t3, t28a, lo, hi);
    c[ 4 * stride] = cadd(t4, t27, lo, hi);
    c[ 5 * stride] = cadd(t5, t26a, lo, hi);
    c[ 6 * stride] = cadd(t6, t25, lo, hi);
    c[ 7 * stride] = cadd(t7, t24a, lo, hi);
    c[ 8 * stride] = cadd(t8, t23a, lo, hi);
    c[ 9 * stride] = cadd(t9, t22, lo, hi);
    c[10 * stride] = cadd(t10, t21a, lo, hi);
    c[11 * stride] = cadd(t11, t20, lo, hi);
    c[12 * stride] = cadd(t12, t19a, lo, hi);
    c[13 * stride] = cadd(t13, t18, lo, hi);
    c[14 * stride] = cadd(t14, t17a, lo, hi);
    c[15 * stride] = cadd(t15, t16, lo, hi);
    c[16 * stride] = csub(t15, t16, lo, hi);
    c[17 * stride] = csub(t14, t17a, lo, hi);
    c[18 * stride] = csub(t13, t18, lo, hi);
    c[19 * stride] = csub(t12, t19a, lo, hi);
    c[20 * stride] = csub(t11, t20, lo, hi);
    c[21 * stride] = csub(t10, t21a, lo, hi);
    c[22 * stride] = csub(t9, t22, lo, hi);
    c[23 * stride] = csub(t8, t23a, lo, hi);
    c[24 * stride] = csub(t7, t24a, lo, hi);
    c[25 * stride] = csub(t6, t25, lo, hi);
    c[26 * stride] = csub(t5, t26a, lo, hi);
    c[27 * stride] = csub(t4, t27, lo, hi);
    c[28 * stride] = csub(t3, t28a, lo, hi);
    c[29 * stride] = csub(t2, t29, lo, hi);
    c[30 * stride] = csub(t1, t30a, lo, hi);
    c[31 * stride] = csub(t0, t31, lo, hi);
}

// Port of inv_dct64_1d_c (the tx64 recursion of src/itx_1d.c is hardcoded:
// only the first 32 inputs are read, all 64 outputs are written).
template <class D>
static inline void hwy_dct64(const D d, hn::VFromD<D> *const c, const int stride,
                             const hn::VFromD<D> lo, const hn::VFromD<D> hi)
{
    using V = hn::VFromD<D>;
    hwy_dct32<D, true>(d, c, stride << 1, lo, hi);

    const V in1  = c[ 1 * stride], in3  = c[ 3 * stride];
    const V in5  = c[ 5 * stride], in7  = c[ 7 * stride];
    const V in9  = c[ 9 * stride], in11 = c[11 * stride];
    const V in13 = c[13 * stride], in15 = c[15 * stride];
    const V in17 = c[17 * stride], in19 = c[19 * stride];
    const V in21 = c[21 * stride], in23 = c[23 * stride];
    const V in25 = c[25 * stride], in27 = c[27 * stride];
    const V in29 = c[29 * stride], in31 = c[31 * stride];

    V t32a = dot1<12, 2048>(d, in1, 101);
    V t33a = dot1<12, 2048>(d, in31, -2824);
    V t34a = dot1<12, 2048>(d, in17, 1660);
    V t35a = dot1<12, 2048>(d, in15, -1474);
    V t36a = dot1<12, 2048>(d, in9, 897);
    V t37a = dot1<12, 2048>(d, in23, -2191);
    V t38a = dot1<12, 2048>(d, in25, 2359);
    V t39a = dot1<12, 2048>(d, in7, -700);
    V t40a = dot1<12, 2048>(d, in5, 501);
    V t41a = dot1<12, 2048>(d, in27, -2520);
    V t42a = dot1<12, 2048>(d, in21, 2019);
    V t43a = dot1<12, 2048>(d, in11, -1092);
    V t44a = dot1<12, 2048>(d, in13, 1285);
    V t45a = dot1<12, 2048>(d, in19, -1842);
    V t46a = dot1<12, 2048>(d, in29, 2675);
    V t47a = dot1<12, 2048>(d, in3, -301);
    V t48a = dot1<12, 2048>(d, in3, 4085);
    V t49a = dot1<12, 2048>(d, in29, 3102);
    V t50a = dot1<12, 2048>(d, in19, 3659);
    V t51a = dot1<12, 2048>(d, in13, 3889);
    V t52a = dot1<12, 2048>(d, in11, 3948);
    V t53a = dot1<12, 2048>(d, in21, 3564);
    V t54a = dot1<12, 2048>(d, in27, 3229);
    V t55a = dot1<12, 2048>(d, in5, 4065);
    V t56a = dot1<12, 2048>(d, in7, 4036);
    V t57a = dot1<12, 2048>(d, in25, 3349);
    V t58a = dot1<12, 2048>(d, in23, 3461);
    V t59a = dot1<12, 2048>(d, in9, 3996);
    V t60a = dot1<12, 2048>(d, in15, 3822);
    V t61a = dot1<12, 2048>(d, in17, 3745);
    V t62a = dot1<12, 2048>(d, in31, 2967);
    V t63a = dot1<12, 2048>(d, in1, 4095);

    V t32 = cadd(t32a, t33a, lo, hi);
    V t33 = csub(t32a, t33a, lo, hi);
    V t34 = csub(t35a, t34a, lo, hi);
    V t35 = cadd(t35a, t34a, lo, hi);
    V t36 = cadd(t36a, t37a, lo, hi);
    V t37 = csub(t36a, t37a, lo, hi);
    V t38 = csub(t39a, t38a, lo, hi);
    V t39 = cadd(t39a, t38a, lo, hi);
    V t40 = cadd(t40a, t41a, lo, hi);
    V t41 = csub(t40a, t41a, lo, hi);
    V t42 = csub(t43a, t42a, lo, hi);
    V t43 = cadd(t43a, t42a, lo, hi);
    V t44 = cadd(t44a, t45a, lo, hi);
    V t45 = csub(t44a, t45a, lo, hi);
    V t46 = csub(t47a, t46a, lo, hi);
    V t47 = cadd(t47a, t46a, lo, hi);
    V t48 = cadd(t48a, t49a, lo, hi);
    V t49 = csub(t48a, t49a, lo, hi);
    V t50 = csub(t51a, t50a, lo, hi);
    V t51 = cadd(t51a, t50a, lo, hi);
    V t52 = cadd(t52a, t53a, lo, hi);
    V t53 = csub(t52a, t53a, lo, hi);
    V t54 = csub(t55a, t54a, lo, hi);
    V t55 = cadd(t55a, t54a, lo, hi);
    V t56 = cadd(t56a, t57a, lo, hi);
    V t57 = csub(t56a, t57a, lo, hi);
    V t58 = csub(t59a, t58a, lo, hi);
    V t59 = cadd(t59a, t58a, lo, hi);
    V t60 = cadd(t60a, t61a, lo, hi);
    V t61 = csub(t60a, t61a, lo, hi);
    V t62 = csub(t63a, t62a, lo, hi);
    V t63 = cadd(t63a, t62a, lo, hi);

    t33a = hn::Sub(dot2<12, 2048>(d, t33, 4096 - 4076, t62, 401), t33);
    t34a = hn::Sub(dot2<12, 2048>(d, t34, -401, t61, 4096 - 4076), t61);
    t37a = dot2<11, 1024>(d, t37, -1299, t58, 1583);
    t38a = dot2<11, 1024>(d, t38, -1583, t57, -1299);
    t41a = hn::Sub(dot2<12, 2048>(d, t41, 4096 - 3612, t54, 1931), t41);
    t42a = hn::Sub(dot2<12, 2048>(d, t42, -1931, t53, 4096 - 3612), t53);
    t45a = hn::Add(dot2<12, 2048>(d, t45, -1189, t50, 3920 - 4096), t50);
    t46a = hn::Sub(dot2<12, 2048>(d, t46, 4096 - 3920, t49, -1189), t46);
    t49a = hn::Add(dot2<12, 2048>(d, t46, -1189, t49, 3920 - 4096), t49);
    t50a = hn::Add(dot2<12, 2048>(d, t45, 3920 - 4096, t50, 1189), t45);
    t53a = hn::Sub(dot2<12, 2048>(d, t42, 4096 - 3612, t53, 1931), t42);
    t54a = hn::Add(dot2<12, 2048>(d, t41, 1931, t54, 3612 - 4096), t54);
    t57a = dot2<11, 1024>(d, t38, -1299, t57, 1583);
    t58a = dot2<11, 1024>(d, t37, 1583, t58, 1299);
    t61a = hn::Sub(dot2<12, 2048>(d, t34, 4096 - 4076, t61, 401), t34);
    t62a = hn::Add(dot2<12, 2048>(d, t33, 401, t62, 4076 - 4096), t62);

    t32a = cadd(t32, t35, lo, hi);
    t33  = cadd(t33a, t34a, lo, hi);
    t34  = csub(t33a, t34a, lo, hi);
    t35a = csub(t32, t35, lo, hi);
    t36a = csub(t39, t36, lo, hi);
    t37  = csub(t38a, t37a, lo, hi);
    t38  = cadd(t38a, t37a, lo, hi);
    t39a = cadd(t39, t36, lo, hi);
    t40a = cadd(t40, t43, lo, hi);
    t41  = cadd(t41a, t42a, lo, hi);
    t42  = csub(t41a, t42a, lo, hi);
    t43a = csub(t40, t43, lo, hi);
    t44a = csub(t47, t44, lo, hi);
    t45  = csub(t46a, t45a, lo, hi);
    t46  = cadd(t46a, t45a, lo, hi);
    t47a = cadd(t47, t44, lo, hi);
    t48a = cadd(t48, t51, lo, hi);
    t49  = cadd(t49a, t50a, lo, hi);
    t50  = csub(t49a, t50a, lo, hi);
    t51a = csub(t48, t51, lo, hi);
    t52a = csub(t55, t52, lo, hi);
    t53  = csub(t54a, t53a, lo, hi);
    t54  = cadd(t54a, t53a, lo, hi);
    t55a = cadd(t55, t52, lo, hi);
    t56a = cadd(t56, t59, lo, hi);
    t57  = cadd(t57a, t58a, lo, hi);
    t58  = csub(t57a, t58a, lo, hi);
    t59a = csub(t56, t59, lo, hi);
    t60a = csub(t63, t60, lo, hi);
    t61  = csub(t62a, t61a, lo, hi);
    t62  = cadd(t62a, t61a, lo, hi);
    t63a = cadd(t63, t60, lo, hi);

    t34a = hn::Sub(dot2<12, 2048>(d, t34, 4096 - 4017, t61, 799), t34);
    t35  = hn::Sub(dot2<12, 2048>(d, t35a, 4096 - 4017, t60a, 799), t35a);
    t36  = hn::Sub(dot2<12, 2048>(d, t36a, -799, t59a, 4096 - 4017), t59a);
    t37a = hn::Sub(dot2<12, 2048>(d, t37, -799, t58, 4096 - 4017), t58);
    t42a = dot2<11, 1024>(d, t42, -1138, t53, 1703);
    t43  = dot2<11, 1024>(d, t43a, -1138, t52a, 1703);
    t44  = dot2<11, 1024>(d, t44a, -1703, t51a, -1138);
    t45a = dot2<11, 1024>(d, t45, -1703, t50, -1138);
    t50a = dot2<11, 1024>(d, t45, -1138, t50, 1703);
    t51  = dot2<11, 1024>(d, t44a, -1138, t51a, 1703);
    t52  = dot2<11, 1024>(d, t43a, 1703, t52a, 1138);
    t53a = dot2<11, 1024>(d, t42, 1703, t53, 1138);
    t58a = hn::Sub(dot2<12, 2048>(d, t37, 4096 - 4017, t58, 799), t37);
    t59  = hn::Sub(dot2<12, 2048>(d, t36a, 4096 - 4017, t59a, 799), t36a);
    t60  = hn::Add(dot2<12, 2048>(d, t35a, 799, t60a, 4017 - 4096), t60a);
    t61a = hn::Add(dot2<12, 2048>(d, t34, 799, t61, 4017 - 4096), t61);

    t32  = cadd(t32a, t39a, lo, hi);
    t33a = cadd(t33, t38, lo, hi);
    t34  = cadd(t34a, t37a, lo, hi);
    t35a = cadd(t35, t36, lo, hi);
    t36a = csub(t35, t36, lo, hi);
    t37  = csub(t34a, t37a, lo, hi);
    t38a = csub(t33, t38, lo, hi);
    t39  = csub(t32a, t39a, lo, hi);
    t40  = csub(t47a, t40a, lo, hi);
    t41a = csub(t46, t41, lo, hi);
    t42  = csub(t45a, t42a, lo, hi);
    t43a = csub(t44, t43, lo, hi);
    t44a = cadd(t44, t43, lo, hi);
    t45  = cadd(t45a, t42a, lo, hi);
    t46a = cadd(t46, t41, lo, hi);
    t47  = cadd(t47a, t40a, lo, hi);
    t48  = cadd(t48a, t55a, lo, hi);
    t49a = cadd(t49, t54, lo, hi);
    t50  = cadd(t50a, t53a, lo, hi);
    t51a = cadd(t51, t52, lo, hi);
    t52a = csub(t51, t52, lo, hi);
    t53  = csub(t50a, t53a, lo, hi);
    t54a = csub(t49, t54, lo, hi);
    t55  = csub(t48a, t55a, lo, hi);
    t56  = csub(t63a, t56a, lo, hi);
    t57a = csub(t62, t57, lo, hi);
    t58  = csub(t61a, t58a, lo, hi);
    t59a = csub(t60, t59, lo, hi);
    t60a = cadd(t60, t59, lo, hi);
    t61  = cadd(t61a, t58a, lo, hi);
    t62a = cadd(t62, t57, lo, hi);
    t63  = cadd(t63a, t56a, lo, hi);

    t36  = hn::Sub(dot2<12, 2048>(d, t36a, 4096 - 3784, t59a, 1567), t36a);
    t37a = hn::Sub(dot2<12, 2048>(d, t37, 4096 - 3784, t58, 1567), t37);
    t38  = hn::Sub(dot2<12, 2048>(d, t38a, 4096 - 3784, t57a, 1567), t38a);
    t39a = hn::Sub(dot2<12, 2048>(d, t39, 4096 - 3784, t56, 1567), t39);
    t40a = hn::Sub(dot2<12, 2048>(d, t40, -1567, t55, 4096 - 3784), t55);
    t41  = hn::Sub(dot2<12, 2048>(d, t41a, -1567, t54a, 4096 - 3784), t54a);
    t42a = hn::Sub(dot2<12, 2048>(d, t42, -1567, t53, 4096 - 3784), t53);
    t43  = hn::Sub(dot2<12, 2048>(d, t43a, -1567, t52a, 4096 - 3784), t52a);
    t52  = hn::Sub(dot2<12, 2048>(d, t43a, 4096 - 3784, t52a, 1567), t43a);
    t53a = hn::Sub(dot2<12, 2048>(d, t42, 4096 - 3784, t53, 1567), t42);
    t54  = hn::Sub(dot2<12, 2048>(d, t41a, 4096 - 3784, t54a, 1567), t41a);
    t55a = hn::Sub(dot2<12, 2048>(d, t40, 4096 - 3784, t55, 1567), t40);
    t56a = hn::Add(dot2<12, 2048>(d, t39, 1567, t56, 3784 - 4096), t56);
    t57  = hn::Add(dot2<12, 2048>(d, t38a, 1567, t57a, 3784 - 4096), t57a);
    t58a = hn::Add(dot2<12, 2048>(d, t37, 1567, t58, 3784 - 4096), t58);
    t59  = hn::Add(dot2<12, 2048>(d, t36a, 1567, t59a, 3784 - 4096), t59a);

    t32a = cadd(t32, t47, lo, hi);
    t33  = cadd(t33a, t46a, lo, hi);
    t34a = cadd(t34, t45, lo, hi);
    t35  = cadd(t35a, t44a, lo, hi);
    t36a = cadd(t36, t43, lo, hi);
    t37  = cadd(t37a, t42a, lo, hi);
    t38a = cadd(t38, t41, lo, hi);
    t39  = cadd(t39a, t40a, lo, hi);
    t40  = csub(t39a, t40a, lo, hi);
    t41a = csub(t38, t41, lo, hi);
    t42  = csub(t37a, t42a, lo, hi);
    t43a = csub(t36, t43, lo, hi);
    t44  = csub(t35a, t44a, lo, hi);
    t45a = csub(t34, t45, lo, hi);
    t46  = csub(t33a, t46a, lo, hi);
    t47a = csub(t32, t47, lo, hi);
    t48a = csub(t63, t48, lo, hi);
    t49  = csub(t62a, t49a, lo, hi);
    t50a = csub(t61, t50, lo, hi);
    t51  = csub(t60a, t51a, lo, hi);
    t52a = csub(t59, t52, lo, hi);
    t53  = csub(t58a, t53a, lo, hi);
    t54a = csub(t57, t54, lo, hi);
    t55  = csub(t56a, t55a, lo, hi);
    t56  = cadd(t56a, t55a, lo, hi);
    t57a = cadd(t57, t54, lo, hi);
    t58  = cadd(t58a, t53a, lo, hi);
    t59a = cadd(t59, t52, lo, hi);
    t60  = cadd(t60a, t51a, lo, hi);
    t61a = cadd(t61, t50, lo, hi);
    t62  = cadd(t62a, t49a, lo, hi);
    t63a = cadd(t63, t48, lo, hi);

    t40a = rot181(d, hn::Sub(t55, t40));
    t41  = rot181(d, hn::Sub(t54a, t41a));
    t42a = rot181(d, hn::Sub(t53, t42));
    t43  = rot181(d, hn::Sub(t52a, t43a));
    t44a = rot181(d, hn::Sub(t51, t44));
    t45  = rot181(d, hn::Sub(t50a, t45a));
    t46a = rot181(d, hn::Sub(t49, t46));
    t47  = rot181(d, hn::Sub(t48a, t47a));
    t48  = rot181(d, hn::Add(t47a, t48a));
    t49a = rot181(d, hn::Add(t46, t49));
    t50  = rot181(d, hn::Add(t45a, t50a));
    t51a = rot181(d, hn::Add(t44, t51));
    t52  = rot181(d, hn::Add(t43a, t52a));
    t53a = rot181(d, hn::Add(t42, t53));
    t54  = rot181(d, hn::Add(t41a, t54a));
    t55a = rot181(d, hn::Add(t40, t55));

    const V t0  = c[ 0 * stride];
    const V t1  = c[ 2 * stride];
    const V t2  = c[ 4 * stride];
    const V t3  = c[ 6 * stride];
    const V t4  = c[ 8 * stride];
    const V t5  = c[10 * stride];
    const V t6  = c[12 * stride];
    const V t7  = c[14 * stride];
    const V t8  = c[16 * stride];
    const V t9  = c[18 * stride];
    const V t10 = c[20 * stride];
    const V t11 = c[22 * stride];
    const V t12 = c[24 * stride];
    const V t13 = c[26 * stride];
    const V t14 = c[28 * stride];
    const V t15 = c[30 * stride];
    const V t16 = c[32 * stride];
    const V t17 = c[34 * stride];
    const V t18 = c[36 * stride];
    const V t19 = c[38 * stride];
    const V t20 = c[40 * stride];
    const V t21 = c[42 * stride];
    const V t22 = c[44 * stride];
    const V t23 = c[46 * stride];
    const V t24 = c[48 * stride];
    const V t25 = c[50 * stride];
    const V t26 = c[52 * stride];
    const V t27 = c[54 * stride];
    const V t28 = c[56 * stride];
    const V t29 = c[58 * stride];
    const V t30 = c[60 * stride];
    const V t31 = c[62 * stride];

    c[ 0 * stride] = cadd(t0, t63a, lo, hi);
    c[ 1 * stride] = cadd(t1, t62, lo, hi);
    c[ 2 * stride] = cadd(t2, t61a, lo, hi);
    c[ 3 * stride] = cadd(t3, t60, lo, hi);
    c[ 4 * stride] = cadd(t4, t59a, lo, hi);
    c[ 5 * stride] = cadd(t5, t58, lo, hi);
    c[ 6 * stride] = cadd(t6, t57a, lo, hi);
    c[ 7 * stride] = cadd(t7, t56, lo, hi);
    c[ 8 * stride] = cadd(t8, t55a, lo, hi);
    c[ 9 * stride] = cadd(t9, t54, lo, hi);
    c[10 * stride] = cadd(t10, t53a, lo, hi);
    c[11 * stride] = cadd(t11, t52, lo, hi);
    c[12 * stride] = cadd(t12, t51a, lo, hi);
    c[13 * stride] = cadd(t13, t50, lo, hi);
    c[14 * stride] = cadd(t14, t49a, lo, hi);
    c[15 * stride] = cadd(t15, t48, lo, hi);
    c[16 * stride] = cadd(t16, t47, lo, hi);
    c[17 * stride] = cadd(t17, t46a, lo, hi);
    c[18 * stride] = cadd(t18, t45, lo, hi);
    c[19 * stride] = cadd(t19, t44a, lo, hi);
    c[20 * stride] = cadd(t20, t43, lo, hi);
    c[21 * stride] = cadd(t21, t42a, lo, hi);
    c[22 * stride] = cadd(t22, t41, lo, hi);
    c[23 * stride] = cadd(t23, t40a, lo, hi);
    c[24 * stride] = cadd(t24, t39, lo, hi);
    c[25 * stride] = cadd(t25, t38a, lo, hi);
    c[26 * stride] = cadd(t26, t37, lo, hi);
    c[27 * stride] = cadd(t27, t36a, lo, hi);
    c[28 * stride] = cadd(t28, t35, lo, hi);
    c[29 * stride] = cadd(t29, t34a, lo, hi);
    c[30 * stride] = cadd(t30, t33, lo, hi);
    c[31 * stride] = cadd(t31, t32a, lo, hi);
    c[32 * stride] = csub(t31, t32a, lo, hi);
    c[33 * stride] = csub(t30, t33, lo, hi);
    c[34 * stride] = csub(t29, t34a, lo, hi);
    c[35 * stride] = csub(t28, t35, lo, hi);
    c[36 * stride] = csub(t27, t36a, lo, hi);
    c[37 * stride] = csub(t26, t37, lo, hi);
    c[38 * stride] = csub(t25, t38a, lo, hi);
    c[39 * stride] = csub(t24, t39, lo, hi);
    c[40 * stride] = csub(t23, t40a, lo, hi);
    c[41 * stride] = csub(t22, t41, lo, hi);
    c[42 * stride] = csub(t21, t42a, lo, hi);
    c[43 * stride] = csub(t20, t43, lo, hi);
    c[44 * stride] = csub(t19, t44a, lo, hi);
    c[45 * stride] = csub(t18, t45, lo, hi);
    c[46 * stride] = csub(t17, t46a, lo, hi);
    c[47 * stride] = csub(t16, t47, lo, hi);
    c[48 * stride] = csub(t15, t48, lo, hi);
    c[49 * stride] = csub(t14, t49a, lo, hi);
    c[50 * stride] = csub(t13, t50, lo, hi);
    c[51 * stride] = csub(t12, t51a, lo, hi);
    c[52 * stride] = csub(t11, t52, lo, hi);
    c[53 * stride] = csub(t10, t53a, lo, hi);
    c[54 * stride] = csub(t9, t54, lo, hi);
    c[55 * stride] = csub(t8, t55a, lo, hi);
    c[56 * stride] = csub(t7, t56, lo, hi);
    c[57 * stride] = csub(t6, t57a, lo, hi);
    c[58 * stride] = csub(t5, t58, lo, hi);
    c[59 * stride] = csub(t4, t59a, lo, hi);
    c[60 * stride] = csub(t3, t60, lo, hi);
    c[61 * stride] = csub(t2, t61a, lo, hi);
    c[62 * stride] = csub(t1, t62, lo, hi);
    c[63 * stride] = csub(t0, t63a, lo, hi);
}

// Square DCT of size N over a batch of transforms, one per lane (dispatch on
// the dav1d tx size, always the non-tx64 variant except for N == 64).
template <int N, class D>
static inline void hwy_dct_1d(const D d, hn::VFromD<D> *const c, const int stride,
                              const hn::VFromD<D> lo, const hn::VFromD<D> hi)
{
    if constexpr (N == 4) {
        hwy_dct4<D, false>(d, c, stride, lo, hi);
    } else if constexpr (N == 8) {
        hwy_dct8<D, false>(d, c, stride, lo, hi);
    } else if constexpr (N == 16) {
        hwy_dct16<D, false>(d, c, stride, lo, hi);
    } else if constexpr (N == 32) {
        hwy_dct32<D, false>(d, c, stride, lo, hi);
    } else {
        hwy_dct64(d, c, stride, lo, hi);
    }
}

// (a*ca + b*cb + c*cc + e*ce + RND) >> SH with wrapping int32 lanes.
template <int SH, int RND, class D>
static inline hn::VFromD<D> dot4(const D d, const hn::VFromD<D> a, const int ca,
                                 const hn::VFromD<D> b, const int cb,
                                 const hn::VFromD<D> c, const int cc,
                                 const hn::VFromD<D> e, const int ce)
{
    return hn::ShiftRight<SH>(hn::Add(hn::Add(hn::Add(hn::Mul(a, hn::Set(d, ca)),
                                                      hn::Mul(b, hn::Set(d, cb))),
                                              hn::Add(hn::Mul(c, hn::Set(d, cc)),
                                                      hn::Mul(e, hn::Set(d, ce)))),
                                      hn::Set(d, RND)));
}

// Port of inv_adst4_1d_internal_c (src/itx_1d.c); FLIP selects the reversed
// output order of inv_flipadst4_1d_c. adst4 has no CLIP calls in C.
template <class D, bool FLIP>
static inline void hwy_adst4(const D d, hn::VFromD<D> *const c, const int stride,
                             const hn::VFromD<D>, const hn::VFromD<D>)
{
    using V = hn::VFromD<D>;
    const V in0 = c[0 * stride], in1 = c[1 * stride];
    const V in2 = c[2 * stride], in3 = c[3 * stride];

    V o[4];
    o[0] = hn::Add(dot4<12, 2048>(d, in0, 1321, in2, 3803 - 4096,
                                     in3, 2482 - 4096, in1, 3344 - 4096),
                   hn::Add(in2, hn::Add(in3, in1)));
    o[1] = hn::Add(dot4<12, 2048>(d, in0, 2482 - 4096, in2, -1321,
                                     in3, -(3803 - 4096), in1, 3344 - 4096),
                   hn::Add(in0, hn::Sub(in1, in3)));
    o[2] = dot1<8, 128>(d, hn::Add(hn::Sub(in0, in2), in3), 209);
    o[3] = hn::Add(dot4<12, 2048>(d, in0, 3803 - 4096, in2, 2482 - 4096,
                                     in3, -1321, in1, -(3344 - 4096)),
                   hn::Sub(hn::Add(in0, in2), in1));

    for (int i = 0; i < 4; i++)
        c[(FLIP ? 3 - i : i) * stride] = o[i];
}

// Port of inv_adst8_1d_internal_c; FLIP selects inv_flipadst8_1d_c.
template <class D, bool FLIP>
static inline void hwy_adst8(const D d, hn::VFromD<D> *const c, const int stride,
                             const hn::VFromD<D> lo, const hn::VFromD<D> hi)
{
    using V = hn::VFromD<D>;
    const V in0 = c[0 * stride], in1 = c[1 * stride];
    const V in2 = c[2 * stride], in3 = c[3 * stride];
    const V in4 = c[4 * stride], in5 = c[5 * stride];
    const V in6 = c[6 * stride], in7 = c[7 * stride];

    const V t0a = hn::Add(dot2<12, 2048>(d, in7, 4076 - 4096, in0, 401), in7);
    const V t1a = hn::Sub(dot2<12, 2048>(d, in7, 401, in0, -(4076 - 4096)), in0);
    const V t2a = hn::Add(dot2<12, 2048>(d, in5, 3612 - 4096, in2, 1931), in5);
    const V t3a = hn::Sub(dot2<12, 2048>(d, in5, 1931, in2, -(3612 - 4096)), in2);
    V t4a = dot2<11, 1024>(d, in3, 1299, in4, 1583);
    V t5a = dot2<11, 1024>(d, in3, 1583, in4, -1299);
    V t6a = hn::Add(dot2<12, 2048>(d, in1, 1189, in6, 3920 - 4096), in6);
    V t7a = hn::Add(dot2<12, 2048>(d, in1, 3920 - 4096, in6, -1189), in1);

    const V t0 = cadd(t0a, t4a, lo, hi);
    const V t1 = cadd(t1a, t5a, lo, hi);
    V t2 = cadd(t2a, t6a, lo, hi);
    V t3 = cadd(t3a, t7a, lo, hi);
    const V t4 = csub(t0a, t4a, lo, hi);
    const V t5 = csub(t1a, t5a, lo, hi);
    V t6 = csub(t2a, t6a, lo, hi);
    V t7 = csub(t3a, t7a, lo, hi);

    t4a = hn::Add(dot2<12, 2048>(d, t4, 3784 - 4096, t5, 1567), t4);
    t5a = hn::Sub(dot2<12, 2048>(d, t4, 1567, t5, -(3784 - 4096)), t5);
    t6a = hn::Add(dot2<12, 2048>(d, t7, 3784 - 4096, t6, -1567), t7);
    t7a = hn::Add(dot2<12, 2048>(d, t7, 1567, t6, 3784 - 4096), t6);

    V o[8];
    o[0] = cadd(t0, t2, lo, hi);
    o[7] = hn::Neg(cadd(t1, t3, lo, hi));
    t2   = csub(t0, t2, lo, hi);
    t3   = csub(t1, t3, lo, hi);
    o[1] = hn::Neg(cadd(t4a, t6a, lo, hi));
    o[6] = cadd(t5a, t7a, lo, hi);
    t6   = csub(t4a, t6a, lo, hi);
    t7   = csub(t5a, t7a, lo, hi);

    o[3] = hn::Neg(rot181(d, hn::Add(t2, t3)));
    o[4] = rot181(d, hn::Sub(t2, t3));
    o[2] = rot181(d, hn::Add(t6, t7));
    o[5] = hn::Neg(rot181(d, hn::Sub(t6, t7)));

    for (int i = 0; i < 8; i++)
        c[(FLIP ? 7 - i : i) * stride] = o[i];
}

// Port of inv_adst16_1d_internal_c; FLIP selects inv_flipadst16_1d_c.
template <class D, bool FLIP>
static inline void hwy_adst16(const D d, hn::VFromD<D> *const c, const int stride,
                              const hn::VFromD<D> lo, const hn::VFromD<D> hi)
{
    using V = hn::VFromD<D>;
    const V in0  = c[ 0 * stride], in1  = c[ 1 * stride];
    const V in2  = c[ 2 * stride], in3  = c[ 3 * stride];
    const V in4  = c[ 4 * stride], in5  = c[ 5 * stride];
    const V in6  = c[ 6 * stride], in7  = c[ 7 * stride];
    const V in8  = c[ 8 * stride], in9  = c[ 9 * stride];
    const V in10 = c[10 * stride], in11 = c[11 * stride];
    const V in12 = c[12 * stride], in13 = c[13 * stride];
    const V in14 = c[14 * stride], in15 = c[15 * stride];

    V t0  = hn::Add(dot2<12, 2048>(d, in15, 4091 - 4096, in0, 201), in15);
    V t1  = hn::Sub(dot2<12, 2048>(d, in15, 201, in0, -(4091 - 4096)), in0);
    V t2  = hn::Add(dot2<12, 2048>(d, in13, 3973 - 4096, in2, 995), in13);
    V t3  = hn::Sub(dot2<12, 2048>(d, in13, 995, in2, -(3973 - 4096)), in2);
    V t4  = hn::Add(dot2<12, 2048>(d, in11, 3703 - 4096, in4, 1751), in11);
    V t5  = hn::Sub(dot2<12, 2048>(d, in11, 1751, in4, -(3703 - 4096)), in4);
    V t6  = dot2<11, 1024>(d, in9, 1645, in6, 1220);
    V t7  = dot2<11, 1024>(d, in9, 1220, in6, -1645);
    V t8  = hn::Add(dot2<12, 2048>(d, in7, 2751, in8, 3035 - 4096), in8);
    V t9  = hn::Add(dot2<12, 2048>(d, in7, 3035 - 4096, in8, -2751), in7);
    V t10 = hn::Add(dot2<12, 2048>(d, in5, 2106, in10, 3513 - 4096), in10);
    V t11 = hn::Add(dot2<12, 2048>(d, in5, 3513 - 4096, in10, -2106), in5);
    V t12 = hn::Add(dot2<12, 2048>(d, in3, 1380, in12, 3857 - 4096), in12);
    V t13 = hn::Add(dot2<12, 2048>(d, in3, 3857 - 4096, in12, -1380), in3);
    V t14 = hn::Add(dot2<12, 2048>(d, in1, 601, in14, 4052 - 4096), in14);
    V t15 = hn::Add(dot2<12, 2048>(d, in1, 4052 - 4096, in14, -601), in1);

    V t0a  = cadd(t0, t8, lo, hi);
    V t1a  = cadd(t1, t9, lo, hi);
    V t2a  = cadd(t2, t10, lo, hi);
    V t3a  = cadd(t3, t11, lo, hi);
    V t4a  = cadd(t4, t12, lo, hi);
    V t5a  = cadd(t5, t13, lo, hi);
    V t6a  = cadd(t6, t14, lo, hi);
    V t7a  = cadd(t7, t15, lo, hi);
    V t8a  = csub(t0, t8, lo, hi);
    V t9a  = csub(t1, t9, lo, hi);
    V t10a = csub(t2, t10, lo, hi);
    V t11a = csub(t3, t11, lo, hi);
    V t12a = csub(t4, t12, lo, hi);
    V t13a = csub(t5, t13, lo, hi);
    V t14a = csub(t6, t14, lo, hi);
    V t15a = csub(t7, t15, lo, hi);

    t8   = hn::Add(dot2<12, 2048>(d, t8a, 4017 - 4096, t9a, 799), t8a);
    t9   = hn::Sub(dot2<12, 2048>(d, t8a, 799, t9a, -(4017 - 4096)), t9a);
    t10  = hn::Add(dot2<12, 2048>(d, t10a, 2276, t11a, 3406 - 4096), t11a);
    t11  = hn::Add(dot2<12, 2048>(d, t10a, 3406 - 4096, t11a, -2276), t10a);
    t12  = hn::Add(dot2<12, 2048>(d, t13a, 4017 - 4096, t12a, -799), t13a);
    t13  = hn::Add(dot2<12, 2048>(d, t13a, 799, t12a, 4017 - 4096), t12a);
    t14  = hn::Sub(dot2<12, 2048>(d, t15a, 2276, t14a, -(3406 - 4096)), t14a);
    t15  = hn::Add(dot2<12, 2048>(d, t15a, 3406 - 4096, t14a, 2276), t15a);

    t0   = cadd(t0a, t4a, lo, hi);
    t1   = cadd(t1a, t5a, lo, hi);
    t2   = cadd(t2a, t6a, lo, hi);
    t3   = cadd(t3a, t7a, lo, hi);
    t4   = csub(t0a, t4a, lo, hi);
    t5   = csub(t1a, t5a, lo, hi);
    t6   = csub(t2a, t6a, lo, hi);
    t7   = csub(t3a, t7a, lo, hi);
    t8a  = cadd(t8, t12, lo, hi);
    t9a  = cadd(t9, t13, lo, hi);
    t10a = cadd(t10, t14, lo, hi);
    t11a = cadd(t11, t15, lo, hi);
    t12a = csub(t8, t12, lo, hi);
    t13a = csub(t9, t13, lo, hi);
    t14a = csub(t10, t14, lo, hi);
    t15a = csub(t11, t15, lo, hi);

    t4a  = hn::Add(dot2<12, 2048>(d, t4, 3784 - 4096, t5, 1567), t4);
    t5a  = hn::Sub(dot2<12, 2048>(d, t4, 1567, t5, -(3784 - 4096)), t5);
    t6a  = hn::Add(dot2<12, 2048>(d, t7, 3784 - 4096, t6, -1567), t7);
    t7a  = hn::Add(dot2<12, 2048>(d, t7, 1567, t6, 3784 - 4096), t6);
    t12  = hn::Add(dot2<12, 2048>(d, t12a, 3784 - 4096, t13a, 1567), t12a);
    t13  = hn::Sub(dot2<12, 2048>(d, t12a, 1567, t13a, -(3784 - 4096)), t13a);
    t14  = hn::Add(dot2<12, 2048>(d, t15a, 3784 - 4096, t14a, -1567), t15a);
    t15  = hn::Add(dot2<12, 2048>(d, t15a, 1567, t14a, 3784 - 4096), t14a);

    V o[16];
    o[ 0] = cadd(t0, t2, lo, hi);
    o[15] = hn::Neg(cadd(t1, t3, lo, hi));
    t2a   = csub(t0, t2, lo, hi);
    t3a   = csub(t1, t3, lo, hi);
    o[ 3] = hn::Neg(cadd(t4a, t6a, lo, hi));
    o[12] = cadd(t5a, t7a, lo, hi);
    t6    = csub(t4a, t6a, lo, hi);
    t7    = csub(t5a, t7a, lo, hi);
    o[ 1] = hn::Neg(cadd(t8a, t10a, lo, hi));
    o[14] = cadd(t9a, t11a, lo, hi);
    t10   = csub(t8a, t10a, lo, hi);
    t11   = csub(t9a, t11a, lo, hi);
    o[ 2] = cadd(t12, t14, lo, hi);
    o[13] = hn::Neg(cadd(t13, t15, lo, hi));
    t14a  = csub(t12, t14, lo, hi);
    t15a  = csub(t13, t15, lo, hi);

    o[ 7] = hn::Neg(rot181(d, hn::Add(t2a, t3a)));
    o[ 8] = rot181(d, hn::Sub(t2a, t3a));
    o[ 4] = rot181(d, hn::Add(t6, t7));
    o[11] = hn::Neg(rot181(d, hn::Sub(t6, t7)));
    o[ 6] = rot181(d, hn::Add(t10, t11));
    o[ 9] = hn::Neg(rot181(d, hn::Sub(t10, t11)));
    o[ 5] = hn::Neg(rot181(d, hn::Add(t14a, t15a)));
    o[10] = rot181(d, hn::Sub(t14a, t15a));

    for (int i = 0; i < 16; i++)
        c[(FLIP ? 15 - i : i) * stride] = o[i];
}

template <int N, bool FLIP, class D>
static inline void hwy_adst(const D d, hn::VFromD<D> *const c, const int stride,
                            const hn::VFromD<D> lo, const hn::VFromD<D> hi)
{
    if constexpr (N == 4) {
        hwy_adst4<D, FLIP>(d, c, stride, lo, hi);
    } else if constexpr (N == 8) {
        hwy_adst8<D, FLIP>(d, c, stride, lo, hi);
    } else {
        hwy_adst16<D, FLIP>(d, c, stride, lo, hi);
    }
}

// Transpose an rows x cols row-major int32 matrix into a cols x rows one, in
// 4x4 blocks; rows and cols are multiples of 4.
static inline void hwy_transpose_i32(const int32_t *const src, const int s_stride,
                                     int32_t *const dst, const int d_stride,
                                     const int rows, const int cols)
{
#if HWY_MAX_BYTES >= 16
    // Exactly 4 int32 lanes; with one 128-bit block the lower/upper halves of
    // Interleave* are unambiguous on every target.
    const hn::CappedTag<int32_t, 4> d4;
    const hn::Repartition<int64_t, decltype(d4)> d64;
    for (int r = 0; r < rows; r += 4) {
        for (int cb = 0; cb < cols; cb += 4) {
            const auto r0 = hn::LoadU(d4, src + (r + 0) * s_stride + cb);
            const auto r1 = hn::LoadU(d4, src + (r + 1) * s_stride + cb);
            const auto r2 = hn::LoadU(d4, src + (r + 2) * s_stride + cb);
            const auto r3 = hn::LoadU(d4, src + (r + 3) * s_stride + cb);
            const auto t0 = hn::InterleaveLower(d4, r0, r1); // r0c0 r1c0 r0c1 r1c1
            const auto t1 = hn::InterleaveUpper(d4, r0, r1); // r0c2 r1c2 r0c3 r1c3
            const auto t2 = hn::InterleaveLower(d4, r2, r3);
            const auto t3 = hn::InterleaveUpper(d4, r2, r3);
            hn::StoreU(hn::BitCast(d4, hn::InterleaveLower(d64, hn::BitCast(d64, t0),
                                                           hn::BitCast(d64, t2))),
                       d4, dst + (cb + 0) * d_stride + r);
            hn::StoreU(hn::BitCast(d4, hn::InterleaveUpper(d64, hn::BitCast(d64, t0),
                                                           hn::BitCast(d64, t2))),
                       d4, dst + (cb + 1) * d_stride + r);
            hn::StoreU(hn::BitCast(d4, hn::InterleaveLower(d64, hn::BitCast(d64, t1),
                                                           hn::BitCast(d64, t3))),
                       d4, dst + (cb + 2) * d_stride + r);
            hn::StoreU(hn::BitCast(d4, hn::InterleaveUpper(d64, hn::BitCast(d64, t1),
                                                           hn::BitCast(d64, t3))),
                       d4, dst + (cb + 3) * d_stride + r);
        }
    }
#else
    for (int r = 0; r < rows; r++)
        for (int cb = 0; cb < cols; cb++)
            dst[cb * d_stride + r] = src[r * s_stride + cb];
#endif
}

// coef (include/common/bitdepth.h): int16_t for 8bpc, int32_t for 16bpc.
template <typename Pixel> struct CoefOf;
template <> struct CoefOf<uint8_t>  { using type = int16_t; };
template <> struct CoefOf<uint16_t> { using type = int32_t; };

template <typename Coef, class D32>
static inline hn::VFromD<D32> hwy_load_coef(const D32 d32, const Coef *const p,
                                            const int n)
{
    const hn::Rebind<Coef, D32> dc;
    const int L = (int) hn::Lanes(d32);
    const auto v = n >= L ? hn::LoadU(dc, p) : hn::LoadN(dc, p, n);
    if constexpr (sizeof(Coef) == 2) {
        return hn::PromoteTo(d32, v);
    } else {
        return v;
    }
}

// Port of inv_txfm_add_c (src/itx_tmpl.c) for square transforms (never
// is_rect2), with 1D types T1 (first pass, over the column-major coeff
// layout) and T2 (second pass). The dav1d coeff layout is column-major
// (coeff[y + x*sh]), so the first pass vectorizes across coeff rows with
// contiguous loads; the round/shift/clip between passes is folded into the
// first pass' stores, and a transpose makes the second pass' loads contiguous
// as well.
enum Tx1d { kTxDct, kTxAdst, kTxFlipAdst };

template <int N, int T, class D>
static inline void hwy_tx1d(const D d, hn::VFromD<D> *const c, const int stride,
                            const hn::VFromD<D> lo, const hn::VFromD<D> hi)
{
    if constexpr (T == kTxDct) {
        hwy_dct_1d<N>(d, c, stride, lo, hi);
    } else if constexpr (T == kTxAdst) {
        hwy_adst<N, false>(d, c, stride, lo, hi);
    } else {
        hwy_adst<N, true>(d, c, stride, lo, hi);
    }
}

template <typename Pixel, int N, int T1, int T2>
static void itx_sq(Pixel *const dst, const ptrdiff_t stride,
                   typename CoefOf<Pixel>::type *const coeff,
                   const int eob, const int bitdepth_max)
{
    using Coef = typename CoefOf<Pixel>::type;
    constexpr int kSh = N < 32 ? N : 32; // sh == sw == imin(w, 32)
    constexpr int kShift = N == 4 ? 0 : N == 8 ? 1 : 2;
    constexpr int kTx = N == 4 ? 0 : N == 8 ? 1 : N == 16 ? 2 : N == 32 ? 3 : 4;
    constexpr bool kDcOnly = T1 == kTxDct && T2 == kTxDct; // has_dconly
    const int rnd = (1 << kShift) >> 1;

    const hn::ScalableTag<int32_t> d;
    const int L = (int) hn::Lanes(d);
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);

    if (kDcOnly && eob == 0) { // eob < has_dconly
        int dc = coeff[0];
        coeff[0] = 0;
        dc = (dc * 181 + 128) >> 8;
        dc = (dc + rnd) >> kShift;
        dc = (dc * 181 + 128 + 2048) >> 12;
        const auto vdc = hn::Set(d, dc);
        const auto vzero = hn::Zero(d);
        const auto vmax = hn::Set(d, bitdepth_max);
        for (int y = 0; y < N; y++) {
            Pixel *const row = dst + y * pxstride;
            for (int x = 0; x < N; x += L) {
                const int n = N - x < L ? N - x : L;
                hwy_store_px(row + x, d,
                             hn::Clamp(hn::Add(hwy_load_px(d, row + x, n), vdc),
                                       vzero, vmax), n);
            }
        }
        return;
    }

    // src/itx_tmpl.c: 8bpc clips to int16; 16bpc derives the ranges from
    // bitdepth_max (unsigned shifts reproduce the C bit patterns).
    int row_clip_min, col_clip_min;
    if constexpr (sizeof(Pixel) == 1) {
        row_clip_min = col_clip_min = INT16_MIN;
    } else {
        row_clip_min = (int) ((unsigned) ~bitdepth_max << 7);
        col_clip_min = (int) ((unsigned) ~bitdepth_max << 5);
    }
    const auto vrlo = hn::Set(d, row_clip_min);
    const auto vrhi = hn::Set(d, ~row_clip_min);
    const auto vclo = hn::Set(d, col_clip_min);
    const auto vchi = hn::Set(d, ~col_clip_min);

    // Rows [0, last] of the coeff columns are transformed; the rest are zero.
    const int last = dav1d_last_nonzero_col_from_eob[kTx][eob];

    int32_t tmp1[N * kSh]; // [x][y]: first pass outputs, round/shift/clipped
    int32_t tmp2[kSh * N]; // [y][x]: tmp1 transposed, second pass input
    hn::VFromD<decltype(d)> v[N];

    const auto vrnd = hn::Set(d, rnd);
    for (int y0 = 0; y0 <= last; y0 += L) {
        const int n = last + 1 - y0 < L ? last + 1 - y0 : L;
        for (int k = 0; k < kSh; k++)
            v[k] = hwy_load_coef(d, coeff + k * kSh + y0, n);
        hwy_tx1d<N, T1>(d, v, 1, vrlo, vrhi);
        for (int k = 0; k < N; k++) {
            const auto t = hn::Clamp(hn::ShiftRightSame(hn::Add(v[k], vrnd), kShift),
                                     vclo, vchi);
            hn::StoreN(t, d, tmp1 + k * kSh + y0, n);
        }
    }
    memset(coeff, 0, sizeof(Coef) * kSh * kSh);
    if (last + 1 < kSh)
        for (int k = 0; k < N; k++)
            memset(tmp1 + k * kSh + last + 1, 0,
                   sizeof(int32_t) * (kSh - last - 1));

    hwy_transpose_i32(tmp1, kSh, tmp2, N, N, kSh);

    const auto v8 = hn::Set(d, 8);
    const auto vzero = hn::Zero(d);
    const auto vmax = hn::Set(d, bitdepth_max);
    for (int x0 = 0; x0 < N; x0 += L) {
        const int n = N - x0 < L ? N - x0 : L;
        for (int k = 0; k < kSh; k++)
            v[k] = hn::LoadN(d, tmp2 + k * N + x0, n);
        hwy_tx1d<N, T2>(d, v, 1, vclo, vchi);
        for (int y = 0; y < N; y++) {
            const auto add = hn::ShiftRight<4>(hn::Add(v[y], v8));
            Pixel *const row = dst + y * pxstride + x0;
            hwy_store_px(row, d,
                         hn::Clamp(hn::Add(hwy_load_px(d, row, n), add),
                                   vzero, vmax), n);
        }
    }
}

#define ITX_FN(n, t1, t2, name, bpc, coef_t, sfx) \
void inv_txfm_add_##name##_##n##x##n##_##sfx(uint##bpc##_t *dst, \
                                             const ptrdiff_t stride, \
                                             coef_t *coeff, \
                                             const int eob HIGHBD_SUFFIX(bpc)) \
{ \
    itx_sq<uint##bpc##_t, n, t1, t2>(dst, stride, coeff, eob, BD_MAX(bpc)); \
}
/* All 9 {DCT,ADST,FLIPADST}^2 combinations for 4/8/16; DCT_DCT also for
 * 32/64. Function name components are in (first-pass, second-pass) order;
 * the enum slot is the swapped name (e.g. fn dct_adst sits at ADST_DCT),
 * see assign_itx_all_fn16 in src/itx_tmpl.c. */
#define ITX_FN_ALL(n, bpc, coef_t, sfx) \
ITX_FN(n, kTxDct,      kTxDct,      dct_dct,           bpc, coef_t, sfx) \
ITX_FN(n, kTxAdst,     kTxDct,      adst_dct,          bpc, coef_t, sfx) \
ITX_FN(n, kTxDct,      kTxAdst,     dct_adst,          bpc, coef_t, sfx) \
ITX_FN(n, kTxAdst,     kTxAdst,     adst_adst,         bpc, coef_t, sfx) \
ITX_FN(n, kTxFlipAdst, kTxDct,      flipadst_dct,      bpc, coef_t, sfx) \
ITX_FN(n, kTxDct,      kTxFlipAdst, dct_flipadst,      bpc, coef_t, sfx) \
ITX_FN(n, kTxFlipAdst, kTxFlipAdst, flipadst_flipadst, bpc, coef_t, sfx) \
ITX_FN(n, kTxAdst,     kTxFlipAdst, adst_flipadst,     bpc, coef_t, sfx) \
ITX_FN(n, kTxFlipAdst, kTxAdst,     flipadst_adst,     bpc, coef_t, sfx)
#define HIGHBD_SUFFIX(bpc)
#define BD_MAX(bpc) 255
ITX_FN_ALL( 4,  8, int16_t, 8bpc)
ITX_FN_ALL( 8,  8, int16_t, 8bpc)
ITX_FN_ALL(16,  8, int16_t, 8bpc)
ITX_FN(32, kTxDct, kTxDct, dct_dct,  8, int16_t, 8bpc)
ITX_FN(64, kTxDct, kTxDct, dct_dct,  8, int16_t, 8bpc)
#undef HIGHBD_SUFFIX
#undef BD_MAX
#define HIGHBD_SUFFIX(bpc) , const int bitdepth_max
#define BD_MAX(bpc) bitdepth_max
ITX_FN_ALL( 4, 16, int32_t, 16bpc)
ITX_FN_ALL( 8, 16, int32_t, 16bpc)
ITX_FN_ALL(16, 16, int32_t, 16bpc)
ITX_FN(32, kTxDct, kTxDct, dct_dct, 16, int32_t, 16bpc)
ITX_FN(64, kTxDct, kTxDct, dct_dct, 16, int32_t, 16bpc)
#undef HIGHBD_SUFFIX
#undef BD_MAX
#undef ITX_FN_ALL
#undef ITX_FN

}  // namespace HWY_NAMESPACE
}  // namespace dav1d

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace dav1d {
#define ITX_EXPORT(n, name, sfx) HWY_EXPORT(inv_txfm_add_##name##_##n##x##n##_##sfx)
#define ITX_EXPORT_ALL(n, sfx) \
ITX_EXPORT(n, dct_dct, sfx); \
ITX_EXPORT(n, adst_dct, sfx); \
ITX_EXPORT(n, dct_adst, sfx); \
ITX_EXPORT(n, adst_adst, sfx); \
ITX_EXPORT(n, flipadst_dct, sfx); \
ITX_EXPORT(n, dct_flipadst, sfx); \
ITX_EXPORT(n, flipadst_flipadst, sfx); \
ITX_EXPORT(n, adst_flipadst, sfx); \
ITX_EXPORT(n, flipadst_adst, sfx);
ITX_EXPORT_ALL( 4,  8bpc)
ITX_EXPORT_ALL( 8,  8bpc)
ITX_EXPORT_ALL(16,  8bpc)
ITX_EXPORT(32, dct_dct, 8bpc);
ITX_EXPORT(64, dct_dct, 8bpc);
ITX_EXPORT_ALL( 4, 16bpc)
ITX_EXPORT_ALL( 8, 16bpc)
ITX_EXPORT_ALL(16, 16bpc)
ITX_EXPORT(32, dct_dct, 16bpc);
ITX_EXPORT(64, dct_dct, 16bpc);
#undef ITX_EXPORT_ALL
#undef ITX_EXPORT
}  // namespace dav1d

namespace {
// Mirror of Dav1dInvTxfmDSPContext (src/itx.h), so that this file does not
// need dav1d's bitdepth-templated C headers. TX_4X4..TX_64X64 are 0..4 and
// DCT_DCT..FLIPADST_ADST are 0..8 (src/levels.h); N_RECT_TX_SIZES = 19,
// N_TX_TYPES_PLUS_LL = 17.
using ItxFn8 = void (*)(uint8_t *, ptrdiff_t, int16_t *, int);
struct ItxDSP8 {
    ItxFn8 itxfm_add[19][17];
};

using ItxFn16 = void (*)(uint16_t *, ptrdiff_t, int32_t *, int, int);
struct ItxDSP16 {
    ItxFn16 itxfm_add[19][17];
};
}  // namespace

namespace dav1d {

/* dav1d assigns each asymmetric pair to the swapped enum slot: the function
 * adst_dct implements enum DCT_ADST, etc. (assign_itx_all_fn16 in
 * src/itx_tmpl.c). */
#define ITX_ASSIGN(tx, n, name, type, sfx) \
    ctx->itxfm_add[tx][type] = HWY_DYNAMIC_POINTER(inv_txfm_add_##name##_##n##x##n##_##sfx)
#define ITX_ASSIGN_ALL(tx, n, sfx) \
    ITX_ASSIGN(tx, n, dct_dct,           0 /*DCT_DCT*/,           sfx); \
    ITX_ASSIGN(tx, n, dct_adst,          1 /*ADST_DCT*/,          sfx); \
    ITX_ASSIGN(tx, n, adst_dct,          2 /*DCT_ADST*/,          sfx); \
    ITX_ASSIGN(tx, n, adst_adst,         3 /*ADST_ADST*/,         sfx); \
    ITX_ASSIGN(tx, n, dct_flipadst,      4 /*FLIPADST_DCT*/,      sfx); \
    ITX_ASSIGN(tx, n, flipadst_dct,      5 /*DCT_FLIPADST*/,      sfx); \
    ITX_ASSIGN(tx, n, flipadst_flipadst, 6 /*FLIPADST_FLIPADST*/, sfx); \
    ITX_ASSIGN(tx, n, flipadst_adst,     7 /*ADST_FLIPADST*/,     sfx); \
    ITX_ASSIGN(tx, n, adst_flipadst,     8 /*FLIPADST_ADST*/,     sfx)

static void itx_dsp_init_8bpc_hwy(void *const c) {
    auto *const ctx = static_cast<ItxDSP8 *>(c);
    ITX_ASSIGN_ALL(0,  4, 8bpc);
    ITX_ASSIGN_ALL(1,  8, 8bpc);
    ITX_ASSIGN_ALL(2, 16, 8bpc);
    ITX_ASSIGN(3, 32, dct_dct, 0, 8bpc);
    ITX_ASSIGN(4, 64, dct_dct, 0, 8bpc);
}

static void itx_dsp_init_16bpc_hwy(void *const c) {
    auto *const ctx = static_cast<ItxDSP16 *>(c);
    ITX_ASSIGN_ALL(0,  4, 16bpc);
    ITX_ASSIGN_ALL(1,  8, 16bpc);
    ITX_ASSIGN_ALL(2, 16, 16bpc);
    ITX_ASSIGN(3, 32, dct_dct, 0, 16bpc);
    ITX_ASSIGN(4, 64, dct_dct, 0, 16bpc);
}

#undef ITX_ASSIGN_ALL
#undef ITX_ASSIGN

}  // namespace dav1d

// bpc only selects 10-bit-specific variants in the asm inits; the Highway
// 16bpc functions cover 10/12 bits via bitdepth_max.
extern "C" void dav1d_itx_dsp_init_hwy_8bpc(void *const c, const int bpc) {
    (void) bpc;
    dav1d::hwy_init_chosen_target();
    dav1d::itx_dsp_init_8bpc_hwy(c);
}

extern "C" void dav1d_itx_dsp_init_hwy_16bpc(void *const c, const int bpc) {
    (void) bpc;
    dav1d::hwy_init_chosen_target();
    dav1d::itx_dsp_init_16bpc_hwy(c);
}

#endif  // HWY_ONCE
