/*
 * Copyright © 2026, VideoLAN and dav1d authors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
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

// Loop restoration filters (src/looprestoration_tmpl.c) implemented with
// Google Highway: one source is compiled per SIMD target and the best one
// supported by the CPU is selected at runtime (HWY_DYNAMIC_DISPATCH).
// Bit-exact with the C code, including the wrapping 32-bit arithmetic that
// the SGR intermediate calculations rely on (reachable for 12bpc content).

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/hwy/looprestoration.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "src/hwy/common.h"

// 1/x table for the SGR "z" values, defined in src/tables.c.
extern "C" const uint8_t dav1d_sgr_x_by_x[256];

// Mirror of LooprestorationParams (src/looprestoration.h); the ALIGN(16) on
// the filter member does not change the offsets. Guarded: this file is
// re-included once per SIMD target.
#ifndef DAV1D_HWY_LRPARAMS
#define DAV1D_HWY_LRPARAMS
union LrParamsM {
    int16_t filter[2][8];
    struct {
        uint32_t s0, s1;
        int16_t w0, w1;
    } sgr;
};
#endif

HWY_BEFORE_NAMESPACE();

namespace dav1d {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// enum LrEdgeFlags in src/looprestoration.h.
enum {
    kHaveLeft   = 1 << 0,
    kHaveRight  = 1 << 1,
    kHaveTop    = 1 << 2,
    kHaveBottom = 1 << 3,
};

// The filters process at most w = 384 columns (w = 256 when LR_HAVE_RIGHT is
// set). SGR row buffers hold w + 2 <= 386 entries, the padded pixel row
// w + 6 <= 390; the slack lets every access use full unaligned vectors
// regardless of the lane count. Only the pixel-buffer loads/stores (which
// must stay inside the w x h rectangle) use counted accesses.
enum {
    kRowStride = 384 + 64,
    kPadRow    = 384 + 6 + 64,
};

// coef (include/common/bitdepth.h): int16_t for 8bpc, int32_t for 16bpc.
template <typename Pixel> struct LrCoefFor;
template <> struct LrCoefFor<uint8_t>  { using type = int16_t; };
template <> struct LrCoefFor<uint16_t> { using type = int32_t; };

// Widens pixel/coef lanes (u8, u16 or i16, values always <= 4095 or within
// the proven coef ranges) to int32 lanes.
template <class D32, class V>
static inline hn::VFromD<D32> hwy_widen_i32(const D32 d32, const V v) {
    if constexpr (sizeof(hn::TFromV<V>) == 4) {
        return v;
    } else if constexpr (sizeof(hn::TFromV<V>) == 2) {
        return hn::PromoteTo(d32, v);
    } else {
        const hn::Rebind<uint16_t, D32> du16;
        return hn::PromoteTo(d32, hn::PromoteTo(du16, v));
    }
}

// coef rows are internal buffers with kRowStride, so stores are always full
// vectors; the narrowing demotion for 8bpc is exact (values in int16 range,
// see the range notes at each producer).
template <typename Coef, class D32>
static inline void hwy_store_coef(Coef *const p, const D32 d32,
                                  const hn::VFromD<D32> v)
{
    if constexpr (sizeof(Coef) == 2) {
        const hn::Rebind<Coef, D32> dc;
        hn::StoreU(hn::DemoteTo(dc, v), dc, p);
    } else {
        hn::StoreU(v, d32, p);
    }
}

template <typename Coef, class D32>
static inline hn::VFromD<D32> hwy_load_coef(const D32 d32, const Coef *const p) {
    if constexpr (sizeof(Coef) == 2) {
        const hn::Rebind<Coef, D32> dc;
        return hn::PromoteTo(d32, hn::LoadU(dc, p));
    } else {
        return hn::LoadU(d32, p);
    }
}

// prow[k] = the source pixel at horizontal position k - 3, with the edge
// extension shared by wiener_filter_h()/sgr_box{3,5}_row_h(): positions < 0
// come from the left buffer (or src[0] without LR_HAVE_LEFT), positions
// >= w from src[w - 1] without LR_HAVE_RIGHT. With LR_HAVE_LEFT but no left
// buffer the C reads src[-3 .. -1] directly; so do we.
template <typename Pixel>
static void hwy_lr_pad_row(uint16_t *const prow, const Pixel (*left)[4],
                           const Pixel *const src, const int w, const int edges)
{
    if (!(edges & kHaveLeft)) {
        prow[0] = prow[1] = prow[2] = (uint16_t) src[0];
    } else if (left) {
        prow[0] = left[0][1];
        prow[1] = left[0][2];
        prow[2] = left[0][3];
    } else {
        prow[0] = (uint16_t) src[-3];
        prow[1] = (uint16_t) src[-2];
        prow[2] = (uint16_t) src[-1];
    }
    const hn::ScalableTag<uint16_t> du16;
    const hn::Rebind<Pixel, decltype(du16)> dp;
    const int LP = (int) hn::Lanes(du16);
    for (int x = 0; x < w; x += LP) {
        const auto v = x + LP <= w ? hn::LoadU(dp, src + x)
                                   : hn::LoadN(dp, src + x, w - x);
        if constexpr (sizeof(Pixel) == 1) {
            hn::StoreU(hn::PromoteTo(du16, v), du16, prow + 3 + x);
        } else {
            hn::StoreU(v, du16, prow + 3 + x);
        }
    }
    if (!(edges & kHaveRight)) {
        prow[w + 3] = prow[w + 4] = prow[w + 5] = (uint16_t) src[w - 1];
    } else {
        prow[w + 3] = (uint16_t) src[w];
        prow[w + 4] = (uint16_t) src[w + 1];
        prow[w + 5] = (uint16_t) src[w + 2];
    }
}

// Port of wiener_filter_h(). The accumulator reaches 1 << (bitdepth + 6) +
// 376 * 4095 for the extreme bitstream-constrained coefficients, and stays
// below 2^30 for arbitrary int16 coefficients; int32 lanes throughout
// (WidenMulAccumulate: widening i16*i16+i32, exact since pixels <= 4095).
template <typename Pixel>
static void hwy_wiener_filter_h(uint16_t *const dst, const Pixel (*left)[4],
                                const Pixel *const src, const int16_t fh[8],
                                const int w, const int edges,
                                const int bitdepth)
{
    const int round_bits_h = 3 + (bitdepth == 12) * 2;
    const int rounding_off_h = 1 << (round_bits_h - 1);
    const int clip_limit = 1 << (bitdepth + 1 + 7 - round_bits_h);

    uint16_t prow[kPadRow];
    hwy_lr_pad_row(prow, left, src, w, edges);

    const hn::ScalableTag<int32_t> d32;
    const hn::Repartition<int16_t, decltype(d32)> d16;
    const hn::Repartition<uint16_t, decltype(d32)> du16;
    const int L = (int) hn::Lanes(d16);
    const auto vbase = hn::Set(d32, 1 << (bitdepth + 6));
    const auto voff = hn::Set(d32, rounding_off_h);
    const auto vzero = hn::Zero(d32);
    const auto vclip = hn::Set(d32, clip_limit - 1);
    for (int x = 0; x < w; x += L) {
        auto lo = vbase, hi = vbase;
        if constexpr (sizeof(Pixel) == 1) {
            const auto p3 = hn::BitCast(d16, hn::LoadU(du16, prow + x + 3));
            lo = hn::WidenMulAccumulate(d32, p3, hn::Set(d16, 128), lo, hi);
        }
        for (int i = 0; i < 7; i++) {
            const auto v = hn::BitCast(d16, hn::LoadU(du16, prow + x + i));
            lo = hn::WidenMulAccumulate(d32, v, hn::Set(d16, fh[i]), lo, hi);
        }
        lo = hn::Clamp(hn::ShiftRightSame(hn::Add(lo, voff), round_bits_h),
                       vzero, vclip);
        hi = hn::Clamp(hn::ShiftRightSame(hn::Add(hi, voff), round_bits_h),
                       vzero, vclip);
        // Values are in [0, 32767]; the u16 demotion is exact.
        hn::StoreU(hn::ReorderDemote2To(du16, lo, hi), du16, dst + x);
    }
}

// Stores the [lo, hi] int32 pair as pixels; the caller has clamped to
// [0, bitdepth_max], so the narrowing demotions are exact.
template <typename Pixel, class D32>
static inline void hwy_store_px2(Pixel *const p,
                                 const hn::VFromD<D32> lo,
                                 const hn::VFromD<D32> hi, const int n)
{
    const hn::Repartition<uint16_t, D32> du16;
    const int L = (int) hn::Lanes(du16);
    if constexpr (sizeof(Pixel) == 1) {
        const hn::Rebind<uint8_t, decltype(du16)> du8;
        const auto v = hn::DemoteTo(du8, hn::ReorderDemote2To(du16, lo, hi));
        if (n >= L) hn::StoreU(v, du8, p); else hn::StoreN(v, du8, p, n);
    } else {
        const auto v = hn::ReorderDemote2To(du16, lo, hi);
        if (n >= L) hn::StoreU(v, du16, p); else hn::StoreN(v, du16, p, n);
    }
}

// Vertical pass over the horizontally filtered rows (u16, <= 32767 after the
// h-pass clip). With arbitrary int16 coefficients the sum can exceed int32;
// the C code overflows its int accumulator the same way (wraps), so the
// wrapping vector arithmetic stays bit-exact in every case.
template <typename Pixel>
static void hwy_wiener_filter_v_core(Pixel *const p,
                                     const uint16_t *const *const r,
                                     const int16_t fv[8], const int w,
                                     const int bitdepth,
                                     const int bitdepth_max)
{
    const int round_bits_v = 11 - (bitdepth == 12) * 2;
    const int rounding_off_v = 1 << (round_bits_v - 1);
    const int round_offset = 1 << (bitdepth + round_bits_v - 1);

    const hn::ScalableTag<int32_t> d32;
    const hn::Repartition<int16_t, decltype(d32)> d16;
    const hn::Repartition<uint16_t, decltype(d32)> du16;
    const int L = (int) hn::Lanes(d16);
    const auto voff = hn::Set(d32, rounding_off_v);
    const auto vzero = hn::Zero(d32);
    const auto vmax = hn::Set(d32, bitdepth_max);
    for (int x = 0; x < w; x += L) {
        auto lo = hn::Set(d32, -round_offset), hi = lo;
        for (int k = 0; k < 7; k++) {
            const auto v = hn::BitCast(d16, hn::LoadU(du16, r[k] + x));
            lo = hn::WidenMulAccumulate(d32, v, hn::Set(d16, fv[k]), lo, hi);
        }
        lo = hn::Clamp(hn::ShiftRightSame(hn::Add(lo, voff), round_bits_v),
                       vzero, vmax);
        hi = hn::Clamp(hn::ShiftRightSame(hn::Add(hi, voff), round_bits_v),
                       vzero, vmax);
        hwy_store_px2<Pixel, decltype(d32)>(p + x, lo, hi, w - x);
    }
}

// Port of wiener_filter_v(): the 7th row is assumed identical to the 6th.
template <typename Pixel>
static void hwy_wiener_filter_v(Pixel *const p, uint16_t **const ptrs,
                                const int16_t fv[8], const int w,
                                const int bitdepth, const int bitdepth_max)
{
    const uint16_t *const r[7] = {
        ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[5]
    };
    hwy_wiener_filter_v_core(p, r, fv, w, bitdepth, bitdepth_max);
    for (int i = 0; i < 5; i++) ptrs[i] = ptrs[i + 1];
}

// Port of wiener_filter_hv(); only [0, w) of tmp is ever read back, so the
// row copy can stop at w (the C code copies the whole stride).
template <typename Pixel>
static void hwy_wiener_filter_hv(Pixel *const p, uint16_t **const ptrs,
                                 const Pixel (*left)[4],
                                 const Pixel *const src,
                                 const int16_t filter[2][8], const int w,
                                 const int edges, const int bitdepth,
                                 const int bitdepth_max)
{
    uint16_t tmp[kRowStride];
    hwy_wiener_filter_h(tmp, left, src, filter[0], w, edges, bitdepth);
    const uint16_t *const r[7] = {
        ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], tmp
    };
    hwy_wiener_filter_v_core(p, r, filter[1], w, bitdepth, bitdepth_max);
    memcpy(ptrs[6], tmp, sizeof(uint16_t) * (size_t) w);
    for (int i = 0; i < 6; i++) ptrs[i] = ptrs[i + 1];
    ptrs[6] = ptrs[0];
}

// Port of wiener_c(); the row-window pointer choreography (including the
// early-exit gotos for small h) is unchanged, only the filter primitives are
// vectorized.
template <typename Pixel>
static void hwy_wiener(Pixel *p, const ptrdiff_t stride,
                       const Pixel (*left)[4], const Pixel *lpf,
                       const int w, int h, const LrParamsM *const params,
                       const int edges, const int bitdepth_max)
{
    const int bitdepth = hwy_ulog2((unsigned) bitdepth_max) + 1;
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);
    uint16_t hor[6 * kRowStride];
    uint16_t *ptrs[7], *rows[6];
    for (int i = 0; i < 6; i++)
        rows[i] = &hor[i * kRowStride];
    const int16_t (*const filter)[8] = params->filter;
    const int16_t *const fh = params->filter[0];
    const int16_t *const fv = params->filter[1];
    const Pixel *lpf_bottom = lpf + 6 * pxstride;

    const Pixel *src = p;
    if (edges & kHaveTop) {
        ptrs[0] = rows[0];
        ptrs[1] = rows[0];
        ptrs[2] = rows[1];
        ptrs[3] = rows[2];
        ptrs[4] = rows[2];
        ptrs[5] = rows[2];

        hwy_wiener_filter_h(rows[0], (const Pixel (*)[4]) nullptr, lpf, fh, w, edges, bitdepth);
        lpf += pxstride;
        hwy_wiener_filter_h(rows[1], (const Pixel (*)[4]) nullptr, lpf, fh, w, edges, bitdepth);

        hwy_wiener_filter_h(rows[2], left, src, fh, w, edges, bitdepth);
        left++;
        src += pxstride;

        if (--h <= 0)
            goto v1;

        ptrs[4] = ptrs[5] = rows[3];
        hwy_wiener_filter_h(rows[3], left, src, fh, w, edges, bitdepth);
        left++;
        src += pxstride;

        if (--h <= 0)
            goto v2;

        ptrs[5] = rows[4];
        hwy_wiener_filter_h(rows[4], left, src, fh, w, edges, bitdepth);
        left++;
        src += pxstride;

        if (--h <= 0)
            goto v3;
    } else {
        ptrs[0] = rows[0];
        ptrs[1] = rows[0];
        ptrs[2] = rows[0];
        ptrs[3] = rows[0];
        ptrs[4] = rows[0];
        ptrs[5] = rows[0];

        hwy_wiener_filter_h(rows[0], left, src, fh, w, edges, bitdepth);
        left++;
        src += pxstride;

        if (--h <= 0)
            goto v1;

        ptrs[4] = ptrs[5] = rows[1];
        hwy_wiener_filter_h(rows[1], left, src, fh, w, edges, bitdepth);
        left++;
        src += pxstride;

        if (--h <= 0)
            goto v2;

        ptrs[5] = rows[2];
        hwy_wiener_filter_h(rows[2], left, src, fh, w, edges, bitdepth);
        left++;
        src += pxstride;

        if (--h <= 0)
            goto v3;

        ptrs[6] = rows[3];
        hwy_wiener_filter_hv(p, ptrs, left, src, filter, w, edges,
                             bitdepth, bitdepth_max);
        left++;
        src += pxstride;
        p += pxstride;

        if (--h <= 0)
            goto v3;

        ptrs[6] = rows[4];
        hwy_wiener_filter_hv(p, ptrs, left, src, filter, w, edges,
                             bitdepth, bitdepth_max);
        left++;
        src += pxstride;
        p += pxstride;

        if (--h <= 0)
            goto v3;
    }

    ptrs[6] = ptrs[5] + kRowStride;
    do {
        hwy_wiener_filter_hv(p, ptrs, left, src, filter, w, edges,
                             bitdepth, bitdepth_max);
        left++;
        src += pxstride;
        p += pxstride;
    } while (--h > 0);

    if (!(edges & kHaveBottom))
        goto v3;

    hwy_wiener_filter_hv(p, ptrs, (const Pixel (*)[4]) nullptr, lpf_bottom, filter, w, edges,
                         bitdepth, bitdepth_max);
    lpf_bottom += pxstride;
    p += pxstride;

    hwy_wiener_filter_hv(p, ptrs, (const Pixel (*)[4]) nullptr, lpf_bottom, filter, w, edges,
                         bitdepth, bitdepth_max);
    p += pxstride;
v1:
    hwy_wiener_filter_v(p, ptrs, fv, w, bitdepth, bitdepth_max);

    return;

v3:
    hwy_wiener_filter_v(p, ptrs, fv, w, bitdepth, bitdepth_max);
    p += pxstride;
v2:
    hwy_wiener_filter_v(p, ptrs, fv, w, bitdepth, bitdepth_max);
    p += pxstride;
    goto v1;
}

// SGR. Ports of the box-filter helpers; the row buffers are indexed like the
// C code (output j corresponds to the C loop index x = j - 1). Window sums
// reach 5 * 4095 and sums of squares 5 * 4095^2 < 2^27 (box5, 12bpc), so
// box sums fit the coef type and everything fits int32.
template <typename Coef>
static void hwy_sgr_box3_sums(int32_t *const sumsq, Coef *const sum,
                              const uint16_t *const prow, const int w)
{
    const hn::ScalableTag<int32_t> d32;
    const hn::Rebind<uint16_t, decltype(d32)> du16;
    const int L = (int) hn::Lanes(d32);
    for (int j = 0; j < w + 2; j += L) {
        const auto p1 = hwy_widen_i32(d32, hn::LoadU(du16, prow + j + 1));
        const auto p2 = hwy_widen_i32(d32, hn::LoadU(du16, prow + j + 2));
        const auto p3 = hwy_widen_i32(d32, hn::LoadU(du16, prow + j + 3));
        hn::StoreU(hn::Add(hn::Add(hn::Mul(p1, p1), hn::Mul(p2, p2)),
                           hn::Mul(p3, p3)), d32, sumsq + j);
        hwy_store_coef(sum + j, d32, hn::Add(hn::Add(p1, p2), p3));
    }
}

template <typename Coef>
static void hwy_sgr_box5_sums(int32_t *const sumsq, Coef *const sum,
                              const uint16_t *const prow, const int w)
{
    const hn::ScalableTag<int32_t> d32;
    const hn::Rebind<uint16_t, decltype(d32)> du16;
    const int L = (int) hn::Lanes(d32);
    for (int j = 0; j < w + 2; j += L) {
        const auto p0 = hwy_widen_i32(d32, hn::LoadU(du16, prow + j + 0));
        const auto p1 = hwy_widen_i32(d32, hn::LoadU(du16, prow + j + 1));
        const auto p2 = hwy_widen_i32(d32, hn::LoadU(du16, prow + j + 2));
        const auto p3 = hwy_widen_i32(d32, hn::LoadU(du16, prow + j + 3));
        const auto p4 = hwy_widen_i32(d32, hn::LoadU(du16, prow + j + 4));
        hn::StoreU(hn::Add(hn::Add(hn::Add(hn::Mul(p0, p0), hn::Mul(p1, p1)),
                                   hn::Add(hn::Mul(p2, p2), hn::Mul(p3, p3))),
                           hn::Mul(p4, p4)), d32, sumsq + j);
        hwy_store_coef(sum + j, d32,
                       hn::Add(hn::Add(hn::Add(p0, p1), hn::Add(p2, p3)), p4));
    }
}

template <typename Pixel, typename Coef>
static void hwy_sgr_box3_row_h(int32_t *const sumsq, Coef *const sum,
                               const Pixel (*left)[4], const Pixel *const src,
                               const int w, const int edges)
{
    uint16_t prow[kPadRow];
    hwy_lr_pad_row(prow, left, src, w, edges);
    hwy_sgr_box3_sums(sumsq, sum, prow, w);
}

template <typename Pixel, typename Coef>
static void hwy_sgr_box5_row_h(int32_t *const sumsq, Coef *const sum,
                               const Pixel (*left)[4], const Pixel *const src,
                               const int w, const int edges)
{
    uint16_t prow[kPadRow];
    hwy_lr_pad_row(prow, left, src, w, edges);
    hwy_sgr_box5_sums(sumsq, sum, prow, w);
}

// Both boxes share one padded row: the box3 window sits at offsets 1..3 of
// the box5 window.
template <typename Pixel, typename Coef>
static void hwy_sgr_box35_row_h(int32_t *const sumsq3, Coef *const sum3,
                                int32_t *const sumsq5, Coef *const sum5,
                                const Pixel (*left)[4], const Pixel *const src,
                                const int w, const int edges)
{
    uint16_t prow[kPadRow];
    hwy_lr_pad_row(prow, left, src, w, edges);
    const hn::ScalableTag<int32_t> d32;
    const hn::Rebind<uint16_t, decltype(d32)> du16;
    const int L = (int) hn::Lanes(d32);
    for (int j = 0; j < w + 2; j += L) {
        const auto p0 = hwy_widen_i32(d32, hn::LoadU(du16, prow + j + 0));
        const auto p1 = hwy_widen_i32(d32, hn::LoadU(du16, prow + j + 1));
        const auto p2 = hwy_widen_i32(d32, hn::LoadU(du16, prow + j + 2));
        const auto p3 = hwy_widen_i32(d32, hn::LoadU(du16, prow + j + 3));
        const auto p4 = hwy_widen_i32(d32, hn::LoadU(du16, prow + j + 4));
        const auto sq1 = hn::Mul(p1, p1);
        const auto sq2 = hn::Mul(p2, p2);
        const auto sq3 = hn::Mul(p3, p3);
        hn::StoreU(hn::Add(hn::Add(sq1, sq2), sq3), d32, sumsq3 + j);
        hwy_store_coef(sum3 + j, d32, hn::Add(hn::Add(p1, p2), p3));
        hn::StoreU(hn::Add(hn::Add(hn::Add(hn::Mul(p0, p0), sq1),
                                   hn::Add(sq2, sq3)),
                           hn::Mul(p4, p4)), d32, sumsq5 + j);
        hwy_store_coef(sum5 + j, d32,
                       hn::Add(hn::Add(hn::Add(p0, p1), hn::Add(p2, p3)), p4));
    }
}

// Vertical box sums over the row buffers; the outputs stay within the same
// ranges (9 or 25 pixels: <= 25 * 4095 and 25 * 4095^2 < 2^29).
template <typename Coef>
static void hwy_sgr_box3_row_v(int32_t **const sumsq, Coef **const sum,
                               int32_t *const sumsq_out, Coef *const sum_out,
                               const int w)
{
    const hn::ScalableTag<int32_t> d32;
    const int L = (int) hn::Lanes(d32);
    for (int x = 0; x < w + 2; x += L) {
        const auto sq = hn::Add(hn::Add(hn::LoadU(d32, sumsq[0] + x),
                                        hn::LoadU(d32, sumsq[1] + x)),
                                hn::LoadU(d32, sumsq[2] + x));
        const auto s = hn::Add(hn::Add(hwy_load_coef(d32, sum[0] + x),
                                       hwy_load_coef(d32, sum[1] + x)),
                               hwy_load_coef(d32, sum[2] + x));
        hn::StoreU(sq, d32, sumsq_out + x);
        hwy_store_coef(sum_out + x, d32, s);
    }
}

template <typename Coef>
static void hwy_sgr_box5_row_v(int32_t **const sumsq, Coef **const sum,
                               int32_t *const sumsq_out, Coef *const sum_out,
                               const int w)
{
    const hn::ScalableTag<int32_t> d32;
    const int L = (int) hn::Lanes(d32);
    for (int x = 0; x < w + 2; x += L) {
        auto sq = hn::Add(hn::LoadU(d32, sumsq[0] + x),
                          hn::LoadU(d32, sumsq[1] + x));
        auto s = hn::Add(hwy_load_coef(d32, sum[0] + x),
                         hwy_load_coef(d32, sum[1] + x));
        for (int k = 2; k < 5; k++) {
            sq = hn::Add(sq, hn::LoadU(d32, sumsq[k] + x));
            s = hn::Add(s, hwy_load_coef(d32, sum[k] + x));
        }
        hn::StoreU(sq, d32, sumsq_out + x);
        hwy_store_coef(sum_out + x, d32, s);
    }
}

// Port of sgr_calc_row_ab(). p * s is computed in uint32 lanes and wraps for
// 12bpc content (p up to ~655M, s up to 3236) exactly like the C unsigned
// arithmetic. x * BB * one_by_x is also unsigned in C (x is unsigned), so
// the >> 12 is a logical shift and AA stays in [0, ~2^20); with BB >= 0 the
// product modulo 2^32 is identical in u32 lanes. The table lookup is a
// scalar gather (z <= 255 after clamping).
template <typename Coef>
static void hwy_sgr_calc_row_ab(int32_t *const AA, Coef *const BB,
                                const int w, const uint32_t s,
                                const int bitdepth_min_8, const int n,
                                const int sgr_one_by_x)
{
    const hn::ScalableTag<int32_t> d32;
    const hn::Rebind<uint32_t, decltype(d32)> du32;
    const int L = (int) hn::Lanes(d32);
    const int cnt = w + 2;
    uint32_t zbuf[kRowStride];
    int32_t xbuf[kRowStride];
    const auto vha = hn::Set(d32, (1 << (2 * bitdepth_min_8)) >> 1);
    const auto vhb = hn::Set(d32, (1 << bitdepth_min_8) >> 1);
    const auto vn = hn::Set(d32, n);
    for (int i = 0; i < cnt; i += L) {
        const auto a = hn::ShiftRightSame(hn::Add(hn::LoadU(d32, AA + i), vha),
                                          2 * bitdepth_min_8);
        const auto b = hn::ShiftRightSame(hn::Add(hwy_load_coef(d32, BB + i),
                                                  vhb), bitdepth_min_8);
        const auto p = hn::Max(hn::Sub(hn::Mul(a, vn), hn::Mul(b, b)),
                               hn::Zero(d32));
        const auto z = hn::Min(hn::ShiftRightSame(hn::Add(
                                   hn::Mul(hn::BitCast(du32, p),
                                           hn::Set(du32, s)),
                                   hn::Set(du32, 1 << 19)), 20),
                               hn::Set(du32, 255));
        hn::StoreU(z, du32, zbuf + i);
    }
    for (int i = 0; i < cnt; i++)
        xbuf[i] = dav1d_sgr_x_by_x[zbuf[i]];
    const auto vone_by_x = hn::Set(du32, (uint32_t) sgr_one_by_x);
    const auto vrnd = hn::Set(du32, 1 << 11);
    for (int i = 0; i < cnt; i += L) {
        const auto x = hn::BitCast(du32, hn::LoadU(d32, xbuf + i));
        const auto bsum = hn::BitCast(du32, hwy_load_coef(d32, BB + i));
        const auto a = hn::ShiftRightSame(hn::Add(hn::Mul(hn::Mul(x, bsum),
                                                          vone_by_x), vrnd), 12);
        hn::StoreU(hn::BitCast(d32, a), d32, AA + i);
        hwy_store_coef(BB + i, d32, hn::BitCast(d32, x)); // x <= 255
    }
}

template <typename Coef>
static void hwy_rotate(int32_t **const sumsq_ptrs, Coef **const sum_ptrs,
                       const int n)
{
    int32_t *const tmp32 = sumsq_ptrs[0];
    Coef *const tmpc = sum_ptrs[0];
    for (int i = 0; i < n - 1; i++) {
        sumsq_ptrs[i] = sumsq_ptrs[i + 1];
        sum_ptrs[i] = sum_ptrs[i + 1];
    }
    sumsq_ptrs[n - 1] = tmp32;
    sum_ptrs[n - 1] = tmpc;
}

template <typename Coef>
static void hwy_rotate5_x2(int32_t **const sumsq_ptrs, Coef **const sum_ptrs)
{
    int32_t *tmp32[2];
    Coef *tmpc[2];
    for (int i = 0; i < 2; i++) {
        tmp32[i] = sumsq_ptrs[i];
        tmpc[i] = sum_ptrs[i];
    }
    for (int i = 0; i < 3; i++) {
        sumsq_ptrs[i] = sumsq_ptrs[i + 2];
        sum_ptrs[i] = sum_ptrs[i + 2];
    }
    for (int i = 0; i < 2; i++) {
        sumsq_ptrs[3 + i] = tmp32[i];
        sum_ptrs[3 + i] = tmpc[i];
    }
}

template <typename Coef>
static void hwy_sgr_box3_vert(int32_t **const sumsq, Coef **const sum,
                              int32_t *const sumsq_out, Coef *const sum_out,
                              const int w, const uint32_t s,
                              const int bitdepth_min_8)
{
    hwy_sgr_box3_row_v(sumsq, sum, sumsq_out, sum_out, w);
    hwy_sgr_calc_row_ab(sumsq_out, sum_out, w, s, bitdepth_min_8, 9, 455);
    hwy_rotate(sumsq, sum, 3);
}

template <typename Coef>
static void hwy_sgr_box5_vert(int32_t **const sumsq, Coef **const sum,
                              int32_t *const sumsq_out, Coef *const sum_out,
                              const int w, const uint32_t s,
                              const int bitdepth_min_8)
{
    hwy_sgr_box5_row_v(sumsq, sum, sumsq_out, sum_out, w);
    hwy_sgr_calc_row_ab(sumsq_out, sum_out, w, s, bitdepth_min_8, 25, 164);
    hwy_rotate5_x2(sumsq, sum);
}

template <typename Pixel, typename Coef>
static void hwy_sgr_box3_hv(int32_t **const sumsq, Coef **const sum,
                            int32_t *const AA, Coef *const BB,
                            const Pixel (*left)[4], const Pixel *const src,
                            const int w, const uint32_t s, const int edges,
                            const int bitdepth_min_8)
{
    hwy_sgr_box3_row_h(sumsq[2], sum[2], left, src, w, edges);
    hwy_sgr_box3_vert(sumsq, sum, AA, BB, w, s, bitdepth_min_8);
}

// EIGHT_NEIGHBORS weighted sums of B (<= 255) and A at column i + 1:
// |a| <= 32 * 255, |a * src| < 2^25, |b| <= 32 * max|A|; the result fits the
// coef type ((b - a * src + 256) >> 9 is within [-4120, 11266] for 8bpc).
template <typename Pixel, typename Coef>
static void hwy_sgr_finish_filter_row1(Coef *const tmp, const Pixel *const src,
                                       int32_t **const A_ptrs,
                                       Coef **const B_ptrs, const int w)
{
    const hn::ScalableTag<int32_t> d32;
    const int L = (int) hn::Lanes(d32);
    const auto v3 = hn::Set(d32, 3);
    const auto vrnd = hn::Set(d32, 1 << 8);
    for (int i = 0; i < w; i += L) {
        const auto Bc1 = hwy_load_coef(d32, B_ptrs[1] + i + 1);
        const auto Bc0 = hwy_load_coef(d32, B_ptrs[0] + i + 1);
        const auto Bc2 = hwy_load_coef(d32, B_ptrs[2] + i + 1);
        const auto Bl1 = hwy_load_coef(d32, B_ptrs[1] + i);
        const auto Br1 = hwy_load_coef(d32, B_ptrs[1] + i + 2);
        const auto Bl0 = hwy_load_coef(d32, B_ptrs[0] + i);
        const auto Bl2 = hwy_load_coef(d32, B_ptrs[2] + i);
        const auto Br0 = hwy_load_coef(d32, B_ptrs[0] + i + 2);
        const auto Br2 = hwy_load_coef(d32, B_ptrs[2] + i + 2);
        const auto a = hn::Add(hn::ShiftLeft<2>(hn::Add(hn::Add(Bc1, Bl1),
                               hn::Add(Br1, hn::Add(Bc0, Bc2)))),
                               hn::Mul(hn::Add(hn::Add(Bl0, Bl2),
                                               hn::Add(Br0, Br2)), v3));
        const auto Ac1 = hn::LoadU(d32, A_ptrs[1] + i + 1);
        const auto Ac0 = hn::LoadU(d32, A_ptrs[0] + i + 1);
        const auto Ac2 = hn::LoadU(d32, A_ptrs[2] + i + 1);
        const auto Al1 = hn::LoadU(d32, A_ptrs[1] + i);
        const auto Ar1 = hn::LoadU(d32, A_ptrs[1] + i + 2);
        const auto Al0 = hn::LoadU(d32, A_ptrs[0] + i);
        const auto Al2 = hn::LoadU(d32, A_ptrs[2] + i);
        const auto Ar0 = hn::LoadU(d32, A_ptrs[0] + i + 2);
        const auto Ar2 = hn::LoadU(d32, A_ptrs[2] + i + 2);
        const auto b = hn::Add(hn::ShiftLeft<2>(hn::Add(hn::Add(Ac1, Al1),
                               hn::Add(Ar1, hn::Add(Ac0, Ac2)))),
                               hn::Mul(hn::Add(hn::Add(Al0, Al2),
                                               hn::Add(Ar0, Ar2)), v3));
        const auto px = hwy_load_px(d32, src + i, w - i);
        const auto t = hn::ShiftRightSame(hn::Add(hn::Sub(b, hn::Mul(a, px)),
                                                  vrnd), 9);
        hwy_store_coef(tmp + i, d32, t);
    }
}

// Port of sgr_finish_filter2(): row 0 uses SIX_NEIGHBORS over rows 0/1,
// row 1 only row 1, with the different rounding constants of the C code.
template <typename Pixel, typename Coef>
static void hwy_sgr_finish_filter2(Coef *tmp, const Pixel *src,
                                   const ptrdiff_t src_stride,
                                   int32_t **const A_ptrs,
                                   Coef **const B_ptrs, const int w,
                                   const int h)
{
    const hn::ScalableTag<int32_t> d32;
    const int L = (int) hn::Lanes(d32);
    const auto v5 = hn::Set(d32, 5);
    const auto v6 = hn::Set(d32, 6);
    for (int i = 0; i < w; i += L) {
        const auto Bc = hn::Add(hwy_load_coef(d32, B_ptrs[0] + i + 1),
                                hwy_load_coef(d32, B_ptrs[1] + i + 1));
        const auto Blr = hn::Add(hn::Add(hwy_load_coef(d32, B_ptrs[0] + i),
                                         hwy_load_coef(d32, B_ptrs[1] + i)),
                                 hn::Add(hwy_load_coef(d32, B_ptrs[0] + i + 2),
                                         hwy_load_coef(d32, B_ptrs[1] + i + 2)));
        const auto Ac = hn::Add(hn::LoadU(d32, A_ptrs[0] + i + 1),
                                hn::LoadU(d32, A_ptrs[1] + i + 1));
        const auto Alr = hn::Add(hn::Add(hn::LoadU(d32, A_ptrs[0] + i),
                                         hn::LoadU(d32, A_ptrs[1] + i)),
                                 hn::Add(hn::LoadU(d32, A_ptrs[0] + i + 2),
                                         hn::LoadU(d32, A_ptrs[1] + i + 2)));
        const auto a = hn::Add(hn::Mul(Bc, v6), hn::Mul(Blr, v5));
        const auto b = hn::Add(hn::Mul(Ac, v6), hn::Mul(Alr, v5));
        const auto px = hwy_load_px(d32, src + i, w - i);
        const auto t = hn::ShiftRightSame(hn::Add(hn::Sub(b, hn::Mul(a, px)),
                                                  hn::Set(d32, 1 << 8)), 9);
        hwy_store_coef(tmp + i, d32, t);
    }
    if (h <= 1)
        return;
    tmp += kRowStride;
    src += src_stride / (ptrdiff_t) sizeof(Pixel);
    const int32_t *const A = &A_ptrs[1][1];
    const Coef *const B = &B_ptrs[1][1];
    for (int i = 0; i < w; i += L) {
        const auto a = hn::Add(hn::Mul(hwy_load_coef(d32, B + i), v6),
                               hn::Mul(hn::Add(hwy_load_coef(d32, B + i - 1),
                                               hwy_load_coef(d32, B + i + 1)),
                                       v5));
        const auto b = hn::Add(hn::Mul(hn::LoadU(d32, A + i), v6),
                               hn::Mul(hn::Add(hn::LoadU(d32, A + i - 1),
                                               hn::LoadU(d32, A + i + 1)),
                                       v5));
        const auto px = hwy_load_px(d32, src + i, w - i);
        const auto t = hn::ShiftRightSame(hn::Add(hn::Sub(b, hn::Mul(a, px)),
                                                  hn::Set(d32, 1 << 7)), 8);
        hwy_store_coef(tmp + i, d32, t);
    }
}

template <typename Pixel, typename Coef>
static void hwy_sgr_weighted_row1(Pixel *const dst, const Coef *const t1,
                                  const int w, const int w1,
                                  const int bitdepth_max)
{
    const hn::ScalableTag<int32_t> d32;
    const int L = (int) hn::Lanes(d32);
    const auto vw1 = hn::Set(d32, w1);
    const auto vrnd = hn::Set(d32, 1 << 10);
    const auto vzero = hn::Zero(d32);
    const auto vmax = hn::Set(d32, bitdepth_max);
    for (int i = 0; i < w; i += L) {
        const auto v = hn::Mul(vw1, hwy_load_coef(d32, t1 + i));
        const auto adj = hn::ShiftRightSame(hn::Add(v, vrnd), 11);
        const auto px = hwy_load_px(d32, dst + i, w - i);
        hwy_store_px(dst + i, d32,
                     hn::Clamp(hn::Add(px, adj), vzero, vmax), w - i);
    }
}

template <typename Pixel, typename Coef>
static void hwy_sgr_weighted2(Pixel *dst, const ptrdiff_t dst_stride,
                              const Coef *t1, const Coef *t2,
                              const int w, const int h,
                              const int w0, const int w1,
                              const int bitdepth_max)
{
    const hn::ScalableTag<int32_t> d32;
    const int L = (int) hn::Lanes(d32);
    const ptrdiff_t pxstride = dst_stride / (ptrdiff_t) sizeof(Pixel);
    const auto vw0 = hn::Set(d32, w0);
    const auto vw1 = hn::Set(d32, w1);
    const auto vrnd = hn::Set(d32, 1 << 10);
    const auto vzero = hn::Zero(d32);
    const auto vmax = hn::Set(d32, bitdepth_max);
    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i += L) {
            const auto v = hn::Add(hn::Mul(vw0, hwy_load_coef(d32, t1 + i)),
                                   hn::Mul(vw1, hwy_load_coef(d32, t2 + i)));
            const auto adj = hn::ShiftRightSame(hn::Add(v, vrnd), 11);
            const auto px = hwy_load_px(d32, dst + i, w - i);
            hwy_store_px(dst + i, d32,
                         hn::Clamp(hn::Add(px, adj), vzero, vmax), w - i);
        }
        dst += pxstride;
        t1 += kRowStride;
        t2 += kRowStride;
    }
}

template <typename Pixel, typename Coef>
static void hwy_sgr_finish1(Pixel **const dst, const ptrdiff_t stride,
                            int32_t **const A_ptrs, Coef **const B_ptrs,
                            const int w, const int w1, const int bitdepth_max)
{
    Coef tmp[kRowStride];

    hwy_sgr_finish_filter_row1(tmp, *dst, A_ptrs, B_ptrs, w);
    hwy_sgr_weighted_row1(*dst, tmp, w, w1, bitdepth_max);
    *dst += stride / (ptrdiff_t) sizeof(Pixel);
    hwy_rotate(A_ptrs, B_ptrs, 3);
}

template <typename Pixel, typename Coef>
static void hwy_sgr_finish2(Pixel **const dst, const ptrdiff_t stride,
                            int32_t **const A_ptrs, Coef **const B_ptrs,
                            const int w, const int h, const int w1,
                            const int bitdepth_max)
{
    Coef tmp[2 * kRowStride];
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);

    hwy_sgr_finish_filter2(tmp, *dst, stride, A_ptrs, B_ptrs, w, h);
    hwy_sgr_weighted_row1(*dst, tmp, w, w1, bitdepth_max);
    *dst += pxstride;
    if (h > 1) {
        hwy_sgr_weighted_row1(*dst, tmp + kRowStride, w, w1, bitdepth_max);
        *dst += pxstride;
    }
    hwy_rotate(A_ptrs, B_ptrs, 2);
}

template <typename Pixel, typename Coef>
static void hwy_sgr_finish_mix(Pixel **const dst, const ptrdiff_t stride,
                               int32_t **const A5_ptrs, Coef **const B5_ptrs,
                               int32_t **const A3_ptrs, Coef **const B3_ptrs,
                               const int w, const int h,
                               const int w0, const int w1,
                               const int bitdepth_max)
{
    Coef tmp5[2 * kRowStride];
    Coef tmp3[2 * kRowStride];
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);

    hwy_sgr_finish_filter2(tmp5, *dst, stride, A5_ptrs, B5_ptrs, w, h);
    hwy_sgr_finish_filter_row1(tmp3, *dst, A3_ptrs, B3_ptrs, w);
    if (h > 1)
        hwy_sgr_finish_filter_row1(tmp3 + kRowStride, *dst + pxstride,
                                   &A3_ptrs[1], &B3_ptrs[1], w);
    hwy_sgr_weighted2(*dst, stride, tmp5, tmp3, w, h, w0, w1, bitdepth_max);
    *dst += h * pxstride;
    hwy_rotate(A5_ptrs, B5_ptrs, 2);
    hwy_rotate(A3_ptrs, B3_ptrs, 4);
}

// Port of sgr_3x3_c(); structure and edge/padding flow unchanged.
template <typename Pixel>
static void hwy_sgr_3x3(Pixel *dst, const ptrdiff_t stride,
                        const Pixel (*left)[4], const Pixel *lpf,
                        const int w, int h, const LrParamsM *const params,
                        const int edges, const int bitdepth_max)
{
    using Coef = typename LrCoefFor<Pixel>::type;
    const int bitdepth_min_8 = hwy_ulog2((unsigned) bitdepth_max) - 7;
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);
    const uint32_t s1 = params->sgr.s1;
    const int w1 = params->sgr.w1;

    int32_t sumsq_buf[kRowStride * 3];
    Coef sum_buf[kRowStride * 3];
    int32_t *sumsq_ptrs[3], *sumsq_rows[3];
    Coef *sum_ptrs[3], *sum_rows[3];
    for (int i = 0; i < 3; i++) {
        sumsq_rows[i] = &sumsq_buf[i * kRowStride];
        sum_rows[i] = &sum_buf[i * kRowStride];
    }

    int32_t A_buf[kRowStride * 3];
    Coef B_buf[kRowStride * 3];
    int32_t *A_ptrs[3];
    Coef *B_ptrs[3];
    for (int i = 0; i < 3; i++) {
        A_ptrs[i] = &A_buf[i * kRowStride];
        B_ptrs[i] = &B_buf[i * kRowStride];
    }
    const Pixel *src = dst;
    const Pixel *lpf_bottom = lpf + 6 * pxstride;

    if (edges & kHaveTop) {
        sumsq_ptrs[0] = sumsq_rows[0];
        sumsq_ptrs[1] = sumsq_rows[1];
        sumsq_ptrs[2] = sumsq_rows[2];
        sum_ptrs[0] = sum_rows[0];
        sum_ptrs[1] = sum_rows[1];
        sum_ptrs[2] = sum_rows[2];

        hwy_sgr_box3_row_h(sumsq_rows[0], sum_rows[0], (const Pixel (*)[4]) nullptr, lpf, w, edges);
        lpf += pxstride;
        hwy_sgr_box3_row_h(sumsq_rows[1], sum_rows[1], (const Pixel (*)[4]) nullptr, lpf, w, edges);

        hwy_sgr_box3_hv(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                        left, src, w, s1, edges, bitdepth_min_8);
        left++;
        src += pxstride;
        hwy_rotate(A_ptrs, B_ptrs, 3);

        if (--h <= 0)
            goto vert_1;

        hwy_sgr_box3_hv(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                        left, src, w, s1, edges, bitdepth_min_8);
        left++;
        src += pxstride;
        hwy_rotate(A_ptrs, B_ptrs, 3);

        if (--h <= 0)
            goto vert_2;
    } else {
        sumsq_ptrs[0] = sumsq_rows[0];
        sumsq_ptrs[1] = sumsq_rows[0];
        sumsq_ptrs[2] = sumsq_rows[0];
        sum_ptrs[0] = sum_rows[0];
        sum_ptrs[1] = sum_rows[0];
        sum_ptrs[2] = sum_rows[0];

        hwy_sgr_box3_row_h(sumsq_rows[0], sum_rows[0], left, src, w, edges);
        left++;
        src += pxstride;

        hwy_sgr_box3_vert(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                          w, s1, bitdepth_min_8);
        hwy_rotate(A_ptrs, B_ptrs, 3);

        if (--h <= 0)
            goto vert_1;

        sumsq_ptrs[2] = sumsq_rows[1];
        sum_ptrs[2] = sum_rows[1];

        hwy_sgr_box3_hv(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                        left, src, w, s1, edges, bitdepth_min_8);
        left++;
        src += pxstride;
        hwy_rotate(A_ptrs, B_ptrs, 3);

        if (--h <= 0)
            goto vert_2;

        sumsq_ptrs[2] = sumsq_rows[2];
        sum_ptrs[2] = sum_rows[2];
    }

    do {
        hwy_sgr_box3_hv(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                        left, src, w, s1, edges, bitdepth_min_8);
        left++;
        src += pxstride;

        hwy_sgr_finish1(&dst, stride, A_ptrs, B_ptrs, w, w1, bitdepth_max);
    } while (--h > 0);

    if (!(edges & kHaveBottom))
        goto vert_2;

    hwy_sgr_box3_hv(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                    (const Pixel (*)[4]) nullptr, lpf_bottom, w, s1, edges, bitdepth_min_8);
    lpf_bottom += pxstride;

    hwy_sgr_finish1(&dst, stride, A_ptrs, B_ptrs, w, w1, bitdepth_max);

    hwy_sgr_box3_hv(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                    (const Pixel (*)[4]) nullptr, lpf_bottom, w, s1, edges, bitdepth_min_8);

    hwy_sgr_finish1(&dst, stride, A_ptrs, B_ptrs, w, w1, bitdepth_max);
    return;

vert_2:
    sumsq_ptrs[2] = sumsq_ptrs[1];
    sum_ptrs[2] = sum_ptrs[1];
    hwy_sgr_box3_vert(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                      w, s1, bitdepth_min_8);

    hwy_sgr_finish1(&dst, stride, A_ptrs, B_ptrs, w, w1, bitdepth_max);

output_1:
    sumsq_ptrs[2] = sumsq_ptrs[1];
    sum_ptrs[2] = sum_ptrs[1];
    hwy_sgr_box3_vert(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                      w, s1, bitdepth_min_8);

    hwy_sgr_finish1(&dst, stride, A_ptrs, B_ptrs, w, w1, bitdepth_max);
    return;

vert_1:
    sumsq_ptrs[2] = sumsq_ptrs[1];
    sum_ptrs[2] = sum_ptrs[1];
    hwy_sgr_box3_vert(sumsq_ptrs, sum_ptrs, A_ptrs[2], B_ptrs[2],
                      w, s1, bitdepth_min_8);
    hwy_rotate(A_ptrs, B_ptrs, 3);
    goto output_1;
}

// Port of sgr_5x5_c(); structure and edge/padding flow unchanged.
template <typename Pixel>
static void hwy_sgr_5x5(Pixel *dst, const ptrdiff_t stride,
                        const Pixel (*left)[4], const Pixel *lpf,
                        const int w, int h, const LrParamsM *const params,
                        const int edges, const int bitdepth_max)
{
    using Coef = typename LrCoefFor<Pixel>::type;
    const int bitdepth_min_8 = hwy_ulog2((unsigned) bitdepth_max) - 7;
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);
    const uint32_t s0 = params->sgr.s0;
    const int w0 = params->sgr.w0;

    int32_t sumsq_buf[kRowStride * 5];
    Coef sum_buf[kRowStride * 5];
    int32_t *sumsq_ptrs[5], *sumsq_rows[5];
    Coef *sum_ptrs[5], *sum_rows[5];
    for (int i = 0; i < 5; i++) {
        sumsq_rows[i] = &sumsq_buf[i * kRowStride];
        sum_rows[i] = &sum_buf[i * kRowStride];
    }

    int32_t A_buf[kRowStride * 2];
    Coef B_buf[kRowStride * 2];
    int32_t *A_ptrs[2];
    Coef *B_ptrs[2];
    for (int i = 0; i < 2; i++) {
        A_ptrs[i] = &A_buf[i * kRowStride];
        B_ptrs[i] = &B_buf[i * kRowStride];
    }
    const Pixel *src = dst;
    const Pixel *lpf_bottom = lpf + 6 * pxstride;

    if (edges & kHaveTop) {
        sumsq_ptrs[0] = sumsq_rows[0];
        sumsq_ptrs[1] = sumsq_rows[0];
        sumsq_ptrs[2] = sumsq_rows[1];
        sumsq_ptrs[3] = sumsq_rows[2];
        sumsq_ptrs[4] = sumsq_rows[3];
        sum_ptrs[0] = sum_rows[0];
        sum_ptrs[1] = sum_rows[0];
        sum_ptrs[2] = sum_rows[1];
        sum_ptrs[3] = sum_rows[2];
        sum_ptrs[4] = sum_rows[3];

        hwy_sgr_box5_row_h(sumsq_rows[0], sum_rows[0], (const Pixel (*)[4]) nullptr, lpf, w, edges);
        lpf += pxstride;
        hwy_sgr_box5_row_h(sumsq_rows[1], sum_rows[1], (const Pixel (*)[4]) nullptr, lpf, w, edges);

        hwy_sgr_box5_row_h(sumsq_rows[2], sum_rows[2], left, src, w, edges);
        left++;
        src += pxstride;

        if (--h <= 0)
            goto vert_1;

        hwy_sgr_box5_row_h(sumsq_rows[3], sum_rows[3], left, src, w, edges);
        left++;
        src += pxstride;
        hwy_sgr_box5_vert(sumsq_ptrs, sum_ptrs, A_ptrs[1], B_ptrs[1],
                          w, s0, bitdepth_min_8);
        hwy_rotate(A_ptrs, B_ptrs, 2);

        if (--h <= 0)
            goto vert_2;

        // ptrs are rotated by 2; both [3] and [4] now point at rows[0]; set
        // one of them to point at the previously unused rows[4].
        sumsq_ptrs[3] = sumsq_rows[4];
        sum_ptrs[3] = sum_rows[4];
    } else {
        sumsq_ptrs[0] = sumsq_rows[0];
        sumsq_ptrs[1] = sumsq_rows[0];
        sumsq_ptrs[2] = sumsq_rows[0];
        sumsq_ptrs[3] = sumsq_rows[0];
        sumsq_ptrs[4] = sumsq_rows[0];
        sum_ptrs[0] = sum_rows[0];
        sum_ptrs[1] = sum_rows[0];
        sum_ptrs[2] = sum_rows[0];
        sum_ptrs[3] = sum_rows[0];
        sum_ptrs[4] = sum_rows[0];

        hwy_sgr_box5_row_h(sumsq_rows[0], sum_rows[0], left, src, w, edges);
        left++;
        src += pxstride;

        if (--h <= 0)
            goto vert_1;

        sumsq_ptrs[4] = sumsq_rows[1];
        sum_ptrs[4] = sum_rows[1];

        hwy_sgr_box5_row_h(sumsq_rows[1], sum_rows[1], left, src, w, edges);
        left++;
        src += pxstride;

        hwy_sgr_box5_vert(sumsq_ptrs, sum_ptrs, A_ptrs[1], B_ptrs[1],
                          w, s0, bitdepth_min_8);
        hwy_rotate(A_ptrs, B_ptrs, 2);

        if (--h <= 0)
            goto vert_2;

        sumsq_ptrs[3] = sumsq_rows[2];
        sumsq_ptrs[4] = sumsq_rows[3];
        sum_ptrs[3] = sum_rows[2];
        sum_ptrs[4] = sum_rows[3];

        hwy_sgr_box5_row_h(sumsq_rows[2], sum_rows[2], left, src, w, edges);
        left++;
        src += pxstride;

        if (--h <= 0)
            goto odd;

        hwy_sgr_box5_row_h(sumsq_rows[3], sum_rows[3], left, src, w, edges);
        left++;
        src += pxstride;

        hwy_sgr_box5_vert(sumsq_ptrs, sum_ptrs, A_ptrs[1], B_ptrs[1],
                          w, s0, bitdepth_min_8);
        hwy_sgr_finish2(&dst, stride, A_ptrs, B_ptrs, w, 2, w0, bitdepth_max);

        if (--h <= 0)
            goto vert_2;

        // ptrs are rotated by 2; both [3] and [4] now point at rows[0]; set
        // one of them to point at the previously unused rows[4].
        sumsq_ptrs[3] = sumsq_rows[4];
        sum_ptrs[3] = sum_rows[4];
    }

    do {
        hwy_sgr_box5_row_h(sumsq_ptrs[3], sum_ptrs[3], left, src, w, edges);
        left++;
        src += pxstride;

        if (--h <= 0)
            goto odd;

        hwy_sgr_box5_row_h(sumsq_ptrs[4], sum_ptrs[4], left, src, w, edges);
        left++;
        src += pxstride;

        hwy_sgr_box5_vert(sumsq_ptrs, sum_ptrs, A_ptrs[1], B_ptrs[1],
                          w, s0, bitdepth_min_8);
        hwy_sgr_finish2(&dst, stride, A_ptrs, B_ptrs, w, 2, w0, bitdepth_max);
    } while (--h > 0);

    if (!(edges & kHaveBottom))
        goto vert_2;

    hwy_sgr_box5_row_h(sumsq_ptrs[3], sum_ptrs[3], (const Pixel (*)[4]) nullptr, lpf_bottom,
                       w, edges);
    lpf_bottom += pxstride;
    hwy_sgr_box5_row_h(sumsq_ptrs[4], sum_ptrs[4], (const Pixel (*)[4]) nullptr, lpf_bottom,
                       w, edges);

output_2:
    hwy_sgr_box5_vert(sumsq_ptrs, sum_ptrs, A_ptrs[1], B_ptrs[1],
                      w, s0, bitdepth_min_8);
    hwy_sgr_finish2(&dst, stride, A_ptrs, B_ptrs, w, 2, w0, bitdepth_max);
    return;

vert_2:
    // Duplicate the last row twice more
    sumsq_ptrs[3] = sumsq_ptrs[2];
    sumsq_ptrs[4] = sumsq_ptrs[2];
    sum_ptrs[3] = sum_ptrs[2];
    sum_ptrs[4] = sum_ptrs[2];
    goto output_2;

odd:
    // Copy the last row as padding once
    sumsq_ptrs[4] = sumsq_ptrs[3];
    sum_ptrs[4] = sum_ptrs[3];

    hwy_sgr_box5_vert(sumsq_ptrs, sum_ptrs, A_ptrs[1], B_ptrs[1],
                      w, s0, bitdepth_min_8);
    hwy_sgr_finish2(&dst, stride, A_ptrs, B_ptrs, w, 2, w0, bitdepth_max);

output_1:
    // Duplicate the last row twice more
    sumsq_ptrs[3] = sumsq_ptrs[2];
    sumsq_ptrs[4] = sumsq_ptrs[2];
    sum_ptrs[3] = sum_ptrs[2];
    sum_ptrs[4] = sum_ptrs[2];

    hwy_sgr_box5_vert(sumsq_ptrs, sum_ptrs, A_ptrs[1], B_ptrs[1],
                      w, s0, bitdepth_min_8);
    // Output only one row
    hwy_sgr_finish2(&dst, stride, A_ptrs, B_ptrs, w, 1, w0, bitdepth_max);
    return;

vert_1:
    // Copy the last row as padding once
    sumsq_ptrs[4] = sumsq_ptrs[3];
    sum_ptrs[4] = sum_ptrs[3];

    hwy_sgr_box5_vert(sumsq_ptrs, sum_ptrs, A_ptrs[1], B_ptrs[1],
                      w, s0, bitdepth_min_8);
    hwy_rotate(A_ptrs, B_ptrs, 2);

    goto output_1;
}

// Port of sgr_mix_c(); structure and edge/padding flow unchanged.
template <typename Pixel>
static void hwy_sgr_mix(Pixel *dst, const ptrdiff_t stride,
                        const Pixel (*left)[4], const Pixel *lpf,
                        const int w, int h, const LrParamsM *const params,
                        const int edges, const int bitdepth_max)
{
    using Coef = typename LrCoefFor<Pixel>::type;
    const int bitdepth_min_8 = hwy_ulog2((unsigned) bitdepth_max) - 7;
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);
    const uint32_t s0 = params->sgr.s0;
    const uint32_t s1 = params->sgr.s1;
    const int w0 = params->sgr.w0;
    const int w1 = params->sgr.w1;

    int32_t sumsq5_buf[kRowStride * 5];
    Coef sum5_buf[kRowStride * 5];
    int32_t *sumsq5_ptrs[5], *sumsq5_rows[5];
    Coef *sum5_ptrs[5], *sum5_rows[5];
    for (int i = 0; i < 5; i++) {
        sumsq5_rows[i] = &sumsq5_buf[i * kRowStride];
        sum5_rows[i] = &sum5_buf[i * kRowStride];
    }
    int32_t sumsq3_buf[kRowStride * 3];
    Coef sum3_buf[kRowStride * 3];
    int32_t *sumsq3_ptrs[3], *sumsq3_rows[3];
    Coef *sum3_ptrs[3], *sum3_rows[3];
    for (int i = 0; i < 3; i++) {
        sumsq3_rows[i] = &sumsq3_buf[i * kRowStride];
        sum3_rows[i] = &sum3_buf[i * kRowStride];
    }

    int32_t A5_buf[kRowStride * 2];
    Coef B5_buf[kRowStride * 2];
    int32_t *A5_ptrs[2];
    Coef *B5_ptrs[2];
    for (int i = 0; i < 2; i++) {
        A5_ptrs[i] = &A5_buf[i * kRowStride];
        B5_ptrs[i] = &B5_buf[i * kRowStride];
    }
    int32_t A3_buf[kRowStride * 4];
    Coef B3_buf[kRowStride * 4];
    int32_t *A3_ptrs[4];
    Coef *B3_ptrs[4];
    for (int i = 0; i < 4; i++) {
        A3_ptrs[i] = &A3_buf[i * kRowStride];
        B3_ptrs[i] = &B3_buf[i * kRowStride];
    }
    const Pixel *src = dst;
    const Pixel *lpf_bottom = lpf + 6 * pxstride;

    if (edges & kHaveTop) {
        sumsq5_ptrs[0] = sumsq5_rows[0];
        sumsq5_ptrs[1] = sumsq5_rows[0];
        sumsq5_ptrs[2] = sumsq5_rows[1];
        sumsq5_ptrs[3] = sumsq5_rows[2];
        sumsq5_ptrs[4] = sumsq5_rows[3];
        sum5_ptrs[0] = sum5_rows[0];
        sum5_ptrs[1] = sum5_rows[0];
        sum5_ptrs[2] = sum5_rows[1];
        sum5_ptrs[3] = sum5_rows[2];
        sum5_ptrs[4] = sum5_rows[3];

        sumsq3_ptrs[0] = sumsq3_rows[0];
        sumsq3_ptrs[1] = sumsq3_rows[1];
        sumsq3_ptrs[2] = sumsq3_rows[2];
        sum3_ptrs[0] = sum3_rows[0];
        sum3_ptrs[1] = sum3_rows[1];
        sum3_ptrs[2] = sum3_rows[2];

        hwy_sgr_box35_row_h(sumsq3_rows[0], sum3_rows[0],
                            sumsq5_rows[0], sum5_rows[0],
                            (const Pixel (*)[4]) nullptr, lpf, w, edges);
        lpf += pxstride;
        hwy_sgr_box35_row_h(sumsq3_rows[1], sum3_rows[1],
                            sumsq5_rows[1], sum5_rows[1],
                            (const Pixel (*)[4]) nullptr, lpf, w, edges);

        hwy_sgr_box35_row_h(sumsq3_rows[2], sum3_rows[2],
                            sumsq5_rows[2], sum5_rows[2],
                            left, src, w, edges);
        left++;
        src += pxstride;

        hwy_sgr_box3_vert(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                          w, s1, bitdepth_min_8);
        hwy_rotate(A3_ptrs, B3_ptrs, 4);

        if (--h <= 0)
            goto vert_1;

        hwy_sgr_box35_row_h(sumsq3_ptrs[2], sum3_ptrs[2],
                            sumsq5_rows[3], sum5_rows[3],
                            left, src, w, edges);
        left++;
        src += pxstride;
        hwy_sgr_box5_vert(sumsq5_ptrs, sum5_ptrs, A5_ptrs[1], B5_ptrs[1],
                          w, s0, bitdepth_min_8);
        hwy_rotate(A5_ptrs, B5_ptrs, 2);
        hwy_sgr_box3_vert(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                          w, s1, bitdepth_min_8);
        hwy_rotate(A3_ptrs, B3_ptrs, 4);

        if (--h <= 0)
            goto vert_2;

        // ptrs are rotated by 2; both [3] and [4] now point at rows[0]; set
        // one of them to point at the previously unused rows[4].
        sumsq5_ptrs[3] = sumsq5_rows[4];
        sum5_ptrs[3] = sum5_rows[4];
    } else {
        sumsq5_ptrs[0] = sumsq5_rows[0];
        sumsq5_ptrs[1] = sumsq5_rows[0];
        sumsq5_ptrs[2] = sumsq5_rows[0];
        sumsq5_ptrs[3] = sumsq5_rows[0];
        sumsq5_ptrs[4] = sumsq5_rows[0];
        sum5_ptrs[0] = sum5_rows[0];
        sum5_ptrs[1] = sum5_rows[0];
        sum5_ptrs[2] = sum5_rows[0];
        sum5_ptrs[3] = sum5_rows[0];
        sum5_ptrs[4] = sum5_rows[0];

        sumsq3_ptrs[0] = sumsq3_rows[0];
        sumsq3_ptrs[1] = sumsq3_rows[0];
        sumsq3_ptrs[2] = sumsq3_rows[0];
        sum3_ptrs[0] = sum3_rows[0];
        sum3_ptrs[1] = sum3_rows[0];
        sum3_ptrs[2] = sum3_rows[0];

        hwy_sgr_box35_row_h(sumsq3_rows[0], sum3_rows[0],
                            sumsq5_rows[0], sum5_rows[0],
                            left, src, w, edges);
        left++;
        src += pxstride;

        hwy_sgr_box3_vert(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                          w, s1, bitdepth_min_8);
        hwy_rotate(A3_ptrs, B3_ptrs, 4);

        if (--h <= 0)
            goto vert_1;

        sumsq5_ptrs[4] = sumsq5_rows[1];
        sum5_ptrs[4] = sum5_rows[1];

        sumsq3_ptrs[2] = sumsq3_rows[1];
        sum3_ptrs[2] = sum3_rows[1];

        hwy_sgr_box35_row_h(sumsq3_rows[1], sum3_rows[1],
                            sumsq5_rows[1], sum5_rows[1],
                            left, src, w, edges);
        left++;
        src += pxstride;

        hwy_sgr_box5_vert(sumsq5_ptrs, sum5_ptrs, A5_ptrs[1], B5_ptrs[1],
                          w, s0, bitdepth_min_8);
        hwy_rotate(A5_ptrs, B5_ptrs, 2);
        hwy_sgr_box3_vert(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                          w, s1, bitdepth_min_8);
        hwy_rotate(A3_ptrs, B3_ptrs, 4);

        if (--h <= 0)
            goto vert_2;

        sumsq5_ptrs[3] = sumsq5_rows[2];
        sumsq5_ptrs[4] = sumsq5_rows[3];
        sum5_ptrs[3] = sum5_rows[2];
        sum5_ptrs[4] = sum5_rows[3];

        sumsq3_ptrs[2] = sumsq3_rows[2];
        sum3_ptrs[2] = sum3_rows[2];

        hwy_sgr_box35_row_h(sumsq3_rows[2], sum3_rows[2],
                            sumsq5_rows[2], sum5_rows[2],
                            left, src, w, edges);
        left++;
        src += pxstride;

        hwy_sgr_box3_vert(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                          w, s1, bitdepth_min_8);
        hwy_rotate(A3_ptrs, B3_ptrs, 4);

        if (--h <= 0)
            goto odd;

        hwy_sgr_box35_row_h(sumsq3_ptrs[2], sum3_ptrs[2],
                            sumsq5_rows[3], sum5_rows[3],
                            left, src, w, edges);
        left++;
        src += pxstride;

        hwy_sgr_box5_vert(sumsq5_ptrs, sum5_ptrs, A5_ptrs[1], B5_ptrs[1],
                          w, s0, bitdepth_min_8);
        hwy_sgr_box3_vert(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                          w, s1, bitdepth_min_8);
        hwy_sgr_finish_mix(&dst, stride, A5_ptrs, B5_ptrs, A3_ptrs, B3_ptrs,
                           w, 2, w0, w1, bitdepth_max);

        if (--h <= 0)
            goto vert_2;

        // ptrs are rotated by 2; both [3] and [4] now point at rows[0]; set
        // one of them to point at the previously unused rows[4].
        sumsq5_ptrs[3] = sumsq5_rows[4];
        sum5_ptrs[3] = sum5_rows[4];
    }

    do {
        hwy_sgr_box35_row_h(sumsq3_ptrs[2], sum3_ptrs[2],
                            sumsq5_ptrs[3], sum5_ptrs[3],
                            left, src, w, edges);
        left++;
        src += pxstride;

        hwy_sgr_box3_vert(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                          w, s1, bitdepth_min_8);
        hwy_rotate(A3_ptrs, B3_ptrs, 4);

        if (--h <= 0)
            goto odd;

        hwy_sgr_box35_row_h(sumsq3_ptrs[2], sum3_ptrs[2],
                            sumsq5_ptrs[4], sum5_ptrs[4],
                            left, src, w, edges);
        left++;
        src += pxstride;

        hwy_sgr_box5_vert(sumsq5_ptrs, sum5_ptrs, A5_ptrs[1], B5_ptrs[1],
                          w, s0, bitdepth_min_8);
        hwy_sgr_box3_vert(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                          w, s1, bitdepth_min_8);
        hwy_sgr_finish_mix(&dst, stride, A5_ptrs, B5_ptrs, A3_ptrs, B3_ptrs,
                           w, 2, w0, w1, bitdepth_max);
    } while (--h > 0);

    if (!(edges & kHaveBottom))
        goto vert_2;

    hwy_sgr_box35_row_h(sumsq3_ptrs[2], sum3_ptrs[2],
                        sumsq5_ptrs[3], sum5_ptrs[3],
                        (const Pixel (*)[4]) nullptr, lpf_bottom, w, edges);
    lpf_bottom += pxstride;
    hwy_sgr_box3_vert(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                      w, s1, bitdepth_min_8);
    hwy_rotate(A3_ptrs, B3_ptrs, 4);

    hwy_sgr_box35_row_h(sumsq3_ptrs[2], sum3_ptrs[2],
                        sumsq5_ptrs[4], sum5_ptrs[4],
                        (const Pixel (*)[4]) nullptr, lpf_bottom, w, edges);

output_2:
    hwy_sgr_box5_vert(sumsq5_ptrs, sum5_ptrs, A5_ptrs[1], B5_ptrs[1],
                      w, s0, bitdepth_min_8);
    hwy_sgr_box3_vert(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                      w, s1, bitdepth_min_8);
    hwy_sgr_finish_mix(&dst, stride, A5_ptrs, B5_ptrs, A3_ptrs, B3_ptrs,
                       w, 2, w0, w1, bitdepth_max);
    return;

vert_2:
    // Duplicate the last row twice more
    sumsq5_ptrs[3] = sumsq5_ptrs[2];
    sumsq5_ptrs[4] = sumsq5_ptrs[2];
    sum5_ptrs[3] = sum5_ptrs[2];
    sum5_ptrs[4] = sum5_ptrs[2];

    sumsq3_ptrs[2] = sumsq3_ptrs[1];
    sum3_ptrs[2] = sum3_ptrs[1];
    hwy_sgr_box3_vert(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                      w, s1, bitdepth_min_8);
    hwy_rotate(A3_ptrs, B3_ptrs, 4);

    sumsq3_ptrs[2] = sumsq3_ptrs[1];
    sum3_ptrs[2] = sum3_ptrs[1];

    goto output_2;

odd:
    // Copy the last row as padding once
    sumsq5_ptrs[4] = sumsq5_ptrs[3];
    sum5_ptrs[4] = sum5_ptrs[3];

    sumsq3_ptrs[2] = sumsq3_ptrs[1];
    sum3_ptrs[2] = sum3_ptrs[1];

    hwy_sgr_box5_vert(sumsq5_ptrs, sum5_ptrs, A5_ptrs[1], B5_ptrs[1],
                      w, s0, bitdepth_min_8);
    hwy_sgr_box3_vert(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                      w, s1, bitdepth_min_8);
    hwy_sgr_finish_mix(&dst, stride, A5_ptrs, B5_ptrs, A3_ptrs, B3_ptrs,
                       w, 2, w0, w1, bitdepth_max);

output_1:
    // Duplicate the last row twice more
    sumsq5_ptrs[3] = sumsq5_ptrs[2];
    sumsq5_ptrs[4] = sumsq5_ptrs[2];
    sum5_ptrs[3] = sum5_ptrs[2];
    sum5_ptrs[4] = sum5_ptrs[2];

    sumsq3_ptrs[2] = sumsq3_ptrs[1];
    sum3_ptrs[2] = sum3_ptrs[1];

    hwy_sgr_box5_vert(sumsq5_ptrs, sum5_ptrs, A5_ptrs[1], B5_ptrs[1],
                      w, s0, bitdepth_min_8);
    hwy_sgr_box3_vert(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                      w, s1, bitdepth_min_8);
    hwy_rotate(A3_ptrs, B3_ptrs, 4);
    // Output only one row
    hwy_sgr_finish_mix(&dst, stride, A5_ptrs, B5_ptrs, A3_ptrs, B3_ptrs,
                       w, 1, w0, w1, bitdepth_max);
    return;

vert_1:
    // Copy the last row as padding once
    sumsq5_ptrs[4] = sumsq5_ptrs[3];
    sum5_ptrs[4] = sum5_ptrs[3];

    sumsq3_ptrs[2] = sumsq3_ptrs[1];
    sum3_ptrs[2] = sum3_ptrs[1];

    hwy_sgr_box5_vert(sumsq5_ptrs, sum5_ptrs, A5_ptrs[1], B5_ptrs[1],
                      w, s0, bitdepth_min_8);
    hwy_rotate(A5_ptrs, B5_ptrs, 2);
    hwy_sgr_box3_vert(sumsq3_ptrs, sum3_ptrs, A3_ptrs[3], B3_ptrs[3],
                      w, s1, bitdepth_min_8);
    hwy_rotate(A3_ptrs, B3_ptrs, 4);

    goto output_1;
}

#define LR_FILTER_FNS(bpc, sfx) \
void wiener_filter_##sfx(uint##bpc##_t *const p, const ptrdiff_t stride, \
                         const uint##bpc##_t (*left)[4], \
                         const uint##bpc##_t *const lpf, const int w, \
                         const int h, const LrParamsM *const params, \
                         const int edges HIGHBD_SUFFIX(bpc)) \
{ \
    hwy_wiener<uint##bpc##_t>(p, stride, left, lpf, w, h, params, edges, \
                              BD_MAX(bpc)); \
} \
void sgr_filter_5x5_##sfx(uint##bpc##_t *const dst, const ptrdiff_t stride, \
                          const uint##bpc##_t (*left)[4], \
                          const uint##bpc##_t *const lpf, const int w, \
                          const int h, const LrParamsM *const params, \
                          const int edges HIGHBD_SUFFIX(bpc)) \
{ \
    hwy_sgr_5x5<uint##bpc##_t>(dst, stride, left, lpf, w, h, params, edges, \
                               BD_MAX(bpc)); \
} \
void sgr_filter_3x3_##sfx(uint##bpc##_t *const dst, const ptrdiff_t stride, \
                          const uint##bpc##_t (*left)[4], \
                          const uint##bpc##_t *const lpf, const int w, \
                          const int h, const LrParamsM *const params, \
                          const int edges HIGHBD_SUFFIX(bpc)) \
{ \
    hwy_sgr_3x3<uint##bpc##_t>(dst, stride, left, lpf, w, h, params, edges, \
                               BD_MAX(bpc)); \
} \
void sgr_filter_mix_##sfx(uint##bpc##_t *const dst, const ptrdiff_t stride, \
                          const uint##bpc##_t (*left)[4], \
                          const uint##bpc##_t *const lpf, const int w, \
                          const int h, const LrParamsM *const params, \
                          const int edges HIGHBD_SUFFIX(bpc)) \
{ \
    hwy_sgr_mix<uint##bpc##_t>(dst, stride, left, lpf, w, h, params, edges, \
                               BD_MAX(bpc)); \
}

#define HIGHBD_SUFFIX(bpc)
#define BD_MAX(bpc) 255
LR_FILTER_FNS(8, 8bpc)
#undef HIGHBD_SUFFIX
#undef BD_MAX
#define HIGHBD_SUFFIX(bpc) , const int bitdepth_max
#define BD_MAX(bpc) bitdepth_max
LR_FILTER_FNS(16, 16bpc)
#undef HIGHBD_SUFFIX
#undef BD_MAX
#undef LR_FILTER_FNS

}  // namespace HWY_NAMESPACE
}  // namespace dav1d

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace dav1d {
HWY_EXPORT(wiener_filter_8bpc);
HWY_EXPORT(sgr_filter_5x5_8bpc);
HWY_EXPORT(sgr_filter_3x3_8bpc);
HWY_EXPORT(sgr_filter_mix_8bpc);
HWY_EXPORT(wiener_filter_16bpc);
HWY_EXPORT(sgr_filter_5x5_16bpc);
HWY_EXPORT(sgr_filter_3x3_16bpc);
HWY_EXPORT(sgr_filter_mix_16bpc);
}  // namespace dav1d

namespace {
// Mirrors of Dav1dLoopRestorationDSPContext (src/looprestoration.h), so that
// this file does not need dav1d's bitdepth-templated C headers.
using LrFn8 = void (*)(uint8_t *, ptrdiff_t, const uint8_t (*)[4],
                       const uint8_t *, int, int, const LrParamsM *, int);
using LrFn16 = void (*)(uint16_t *, ptrdiff_t, const uint16_t (*)[4],
                        const uint16_t *, int, int, const LrParamsM *, int,
                        int);
struct LrDSP8 {
    LrFn8 wiener[2];
    LrFn8 sgr[3];
};
struct LrDSP16 {
    LrFn16 wiener[2];
    LrFn16 sgr[3];
};
}  // namespace

namespace dav1d {

static void loop_restoration_dsp_init_8bpc_hwy(void *const c) {
    auto *const ctx = static_cast<LrDSP8 *>(c);
    ctx->wiener[0] = ctx->wiener[1] = HWY_DYNAMIC_POINTER(wiener_filter_8bpc);
    ctx->sgr[0] = HWY_DYNAMIC_POINTER(sgr_filter_5x5_8bpc);
    ctx->sgr[1] = HWY_DYNAMIC_POINTER(sgr_filter_3x3_8bpc);
    ctx->sgr[2] = HWY_DYNAMIC_POINTER(sgr_filter_mix_8bpc);
}

static void loop_restoration_dsp_init_16bpc_hwy(void *const c) {
    auto *const ctx = static_cast<LrDSP16 *>(c);
    ctx->wiener[0] = ctx->wiener[1] = HWY_DYNAMIC_POINTER(wiener_filter_16bpc);
    ctx->sgr[0] = HWY_DYNAMIC_POINTER(sgr_filter_5x5_16bpc);
    ctx->sgr[1] = HWY_DYNAMIC_POINTER(sgr_filter_3x3_16bpc);
    ctx->sgr[2] = HWY_DYNAMIC_POINTER(sgr_filter_mix_16bpc);
}

}  // namespace dav1d

extern "C" void dav1d_loop_restoration_dsp_init_hwy_8bpc(void *const c) {
    dav1d::hwy_init_chosen_target();
    dav1d::loop_restoration_dsp_init_8bpc_hwy(c);
}

extern "C" void dav1d_loop_restoration_dsp_init_hwy_16bpc(void *const c) {
    dav1d::hwy_init_chosen_target();
    dav1d::loop_restoration_dsp_init_16bpc_hwy(c);
}

#endif  // HWY_ONCE
