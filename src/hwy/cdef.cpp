/*
 * Copyright © 2026, VideoLAN and dav1d authors
 * Copyright © 2026, Zhou Qiankang <wszqkzqk@qq.com>
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

// CDEF filter (src/cdef_tmpl.c) implemented with Google Highway: one source
// is compiled per SIMD target and the best one supported by the CPU is
// selected at runtime (HWY_DYNAMIC_DISPATCH). Bit-exact with the C code.

#include <stddef.h>
#include <stdint.h>
#include <type_traits>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/hwy/cdef.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "src/hwy/common.h"

// Flattened (y * 12 + x) neighbour offsets, defined in src/tables.c.
extern "C" const int8_t dav1d_cdef_directions[12][2];

HWY_BEFORE_NAMESPACE();

namespace dav1d {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// enum CdefEdgeFlags in src/cdef.h.
enum {
    kHaveLeft  = 1 << 0,
    kHaveRight = 1 << 1,
    kHaveTop   = 1 << 2,
    kHaveBottom = 1 << 3,
};

static void hwy_fill(int16_t *tmp, const ptrdiff_t stride,
                     const int w, const int h)
{
    if (w == 2) {
        const hn::CappedTag<int16_t, 2> d2;
        const auto vpad = hn::Set(d2, INT16_MIN);
        for (int y = 0; y < h; y++, tmp += stride) hn::StoreU(vpad, d2, tmp);
    } else { // w is 8 or 12
        const hn::CappedTag<int16_t, 8> d8;
        const auto vpad = hn::Set(d8, INT16_MIN);
        for (int y = 0; y < h; y++, tmp += stride) {
            hn::StoreU(vpad, d8, tmp);
            if (w > 8) hn::StoreU(vpad, d8, tmp + w - 8); // overlapping
        }
    }
}

// Widens loaded pixels to int16 lanes (pixel values <= 4095).
template <class D, class V>
static inline hn::VFromD<D> hwy_widen(const D d, const V v) {
    if constexpr (sizeof(hn::TFromV<V>) == 1) {
        return hn::PromoteTo(d, v);
    } else {
        return hn::BitCast(d, v);
    }
}

// Copies n pixels (n in [4, 12]) from src to a tmp row, widening to int16.
// Overlapping end chunks keep every access in the [0, n) range the C code
// touches.
template <typename Pixel>
static inline void hwy_copy_row(int16_t *const dst, const Pixel *const src,
                                const int n)
{
    const hn::CappedTag<Pixel, 8> dp;
    const hn::CappedTag<int16_t, 8> di;
    if (n >= 8) {
        hn::StoreU(hwy_widen(di, hn::LoadU(dp, src)), di, dst);
        if (n > 8)
            hn::StoreU(hwy_widen(di, hn::LoadU(dp, src + n - 8)), di,
                       dst + n - 8);
    } else {
        const hn::Half<decltype(dp)> dhp;
        const hn::Half<decltype(di)> dhi;
        hn::StoreU(hwy_widen(dhi, hn::LoadU(dhp, src)), dhi, dst);
        if (n > 4)
            hn::StoreU(hwy_widen(dhi, hn::LoadU(dhp, src + n - 4)), dhi,
                       dst + n - 4);
    }
}

// Port of padding() from src/cdef_tmpl.c.
template <typename Pixel>
static void hwy_padding(int16_t *tmp, const ptrdiff_t tmp_stride,
                        const Pixel *src, const ptrdiff_t src_stride,
                        const Pixel (*left)[2],
                        const Pixel *top, const Pixel *bottom,
                        const int w, const int h, const int edges)
{
    const ptrdiff_t src_pxstride = src_stride / (ptrdiff_t) sizeof(Pixel);
    int x_start = -2, x_end = w + 2, y_start = -2, y_end = h + 2;
    if (!(edges & kHaveTop)) {
        hwy_fill(tmp - 2 - 2 * tmp_stride, tmp_stride, w + 4, 2);
        y_start = 0;
    }
    if (!(edges & kHaveBottom)) {
        hwy_fill(tmp + h * tmp_stride - 2, tmp_stride, w + 4, 2);
        y_end -= 2;
    }
    if (!(edges & kHaveLeft)) {
        hwy_fill(tmp + y_start * tmp_stride - 2, tmp_stride, 2, y_end - y_start);
        x_start = 0;
    }
    if (!(edges & kHaveRight)) {
        hwy_fill(tmp + y_start * tmp_stride + w, tmp_stride, 2, y_end - y_start);
        x_end -= 2;
    }

    for (int y = y_start; y < 0; y++) {
        hwy_copy_row(tmp + x_start + y * tmp_stride, top + x_start,
                     x_end - x_start);
        top += src_pxstride;
    }
    if (x_start) { // left edge present: copy the two left columns
        const hn::CappedTag<Pixel, 2> dp;
        const hn::CappedTag<int16_t, 2> di;
        for (int y = 0; y < h; y++)
            hn::StoreU(hwy_widen(di, hn::LoadU(dp, left[y])), di,
                       tmp - 2 + y * tmp_stride);
    }
    for (int y = 0; y < h; y++) {
        hwy_copy_row(tmp, src, x_end);
        src += src_pxstride;
        tmp += tmp_stride;
    }
    for (int y = h; y < y_end; y++) {
        hwy_copy_row(tmp + x_start, bottom + x_start, x_end - x_start);
        bottom += src_pxstride;
        tmp += tmp_stride;
    }
}

// One block row per vector; LoadN/StoreN keep lanes [W, Lanes) inert (every
// step is per-lane). All arithmetic is 16-bit: |diff| <= 4095 for real
// pixels, |sum| <= 6240.
template <typename Pixel, int W, int H>
static void cdef_filter_block(Pixel *dst, const ptrdiff_t dst_stride,
                              const Pixel (*left)[2],
                              const Pixel *const top, const Pixel *const bottom,
                              const int pri_strength, const int sec_strength,
                              const int dir, const int damping, const int edges,
                              const int bitdepth_min_8)
{
    const ptrdiff_t tmp_stride = 12;
    int16_t tmp_buf[144];
    int16_t *tmp = tmp_buf + 2 * tmp_stride + 2;
    const int8_t (*const cdef_dirs)[2] = &dav1d_cdef_directions[dir];
    const ptrdiff_t dst_pxstride = dst_stride / (ptrdiff_t) sizeof(Pixel);

    hwy_padding(tmp, tmp_stride, dst, dst_stride, left, top, bottom,
                W, H, edges);

    const hn::ScalableTag<int16_t> d;
    const hn::Rebind<uint16_t, decltype(d)> du16;
    const hn::Rebind<Pixel, decltype(d)> dpix;

    // Pixel values never exceed 4095, so BitCast u16->i16 is safe.
    const auto load_px = [&](const Pixel *const p) {
        if constexpr (sizeof(Pixel) == 1) {
            return hn::BitCast(d, hn::PromoteTo(du16, hn::LoadN(dpix, p, W)));
        } else {
            return hn::BitCast(d, hn::LoadN(dpix, p, W));
        }
    };
    const auto load_tmp = [&](const int16_t *const trow, const int off) {
        return hn::LoadN(d, trow + off, W);
    };

    // constrain() from src/cdef_tmpl.c, in 16-bit: apply_sign(min(adiff, t),
    // diff) == clamp(diff, -t, t) for t >= 0. For INT16_MIN padding lanes,
    // BitCast(Abs(diff)) is a huge u16 value (>= 28673), so the saturating
    // subtraction yields t == 0 exactly like the C code for every legal
    // (threshold, shift).
    const auto constrain = [&](const hn::Vec<decltype(d)> diff,
                               const hn::Vec<decltype(du16)> vthreshold,
                               const int shift) {
        const auto adiff = hn::BitCast(du16, hn::Abs(diff));
        const auto t = hn::BitCast(d, hn::SaturatedSub(vthreshold,
                                        hn::ShiftRightSame(adiff, shift)));
        return hn::Clamp(diff, hn::Neg(t), t);
    };

    const auto v8 = hn::Set(d, 8);
    const auto v1 = hn::Set(d, 1);

    // Rounding: (sum - (sum < 0) + 8) >> 4; px + adj stays within int16.
    const auto round = [&](const hn::Vec<decltype(d)> sum) {
        return hn::ShiftRight<4>(hn::Sub(hn::Add(sum, v8),
                                         hn::And(hn::BroadcastSignBit(sum), v1)));
    };

    const auto store_px = [&](Pixel *const p, const hn::Vec<decltype(d)> v) {
        if constexpr (sizeof(Pixel) == 1) {
            hn::StoreN(hn::DemoteTo(dpix, v), dpix, p, W);
        } else {
            hn::StoreN(hn::BitCast(dpix, v), dpix, p, W);
        }
    };
    // Narrowing stores replicate the C implicit conversions: truncating for
    // the unclamped paths (the 16bpc bitcast already is); the clamped path
    // is in [0, bitdepth_max], so demotion/bitcast are exact.
    const auto store_unclamped = [&](Pixel *const p, hn::Vec<decltype(d)> res) {
        if constexpr (sizeof(Pixel) == 1)
            res = hn::And(res, hn::Set(d, 0xff));
        store_px(p, res);
    };
    const auto store_clamped = [&](Pixel *const p,
                                   hn::Vec<decltype(d)> res,
                                   const hn::Vec<decltype(du16)> vmin,
                                   const hn::Vec<decltype(d)> vmax) {
        res = hn::Min(hn::Max(res, hn::BitCast(d, vmin)), vmax);
        store_px(p, res);
    };

    if (pri_strength) {
        const int pri_tap = 4 - ((pri_strength >> bitdepth_min_8) & 1);
        const int pri_shift = hwy_imax(0, damping - hwy_ulog2(pri_strength));
        const hn::Vec<decltype(d)> vtap[2] = {
            hn::Set(d, pri_tap), hn::Set(d, (pri_tap & 3) | 2) };
        const int sec_shift = damping - hwy_ulog2(sec_strength);
        const auto vpri = hn::BitCast(du16, hn::Set(d, pri_strength));
        const auto vsec = hn::BitCast(du16, hn::Set(d, sec_strength));
        const auto row_loop = [&](auto clamp_c) {
            constexpr bool kClamp = decltype(clamp_c)::value;
            Pixel *drow = dst;
            const int16_t *trow = tmp;
            for (int y = 0; y < H; y++, drow += dst_pxstride, trow += tmp_stride) {
                const auto px = load_px(drow);
                auto sum = hn::Zero(d);
                auto vmin = hn::BitCast(du16, px);
                auto vmax = px;
                // min/max as in the C code: unsigned min / signed max over
                // the raw int16 samples (INT16_MIN padding is huge unsigned
                // and tiny signed, so it is never selected).
                const auto minmax = [&](const hn::Vec<decltype(d)> p0,
                                        const hn::Vec<decltype(d)> p1,
                                        const hn::Vec<decltype(d)> s0,
                                        const hn::Vec<decltype(d)> s1,
                                        const hn::Vec<decltype(d)> s2,
                                        const hn::Vec<decltype(d)> s3) {
                    const auto mx = hn::Max(hn::Max(p0, p1),
                                            hn::Max(hn::Max(s0, s1), hn::Max(s2, s3)));
                    const auto mn = hn::Min(hn::Min(hn::BitCast(du16, p0), hn::BitCast(du16, p1)),
                                            hn::Min(hn::Min(hn::BitCast(du16, s0), hn::BitCast(du16, s1)),
                                                    hn::Min(hn::BitCast(du16, s2), hn::BitCast(du16, s3))));
                    vmax = hn::Max(vmax, mx);
                    vmin = hn::Min(vmin, mn);
                };
                for (int k = 0; k < 2; k++) {
                    const int off1 = cdef_dirs[2][k]; // dir
                    const auto p0 = load_tmp(trow, off1);
                    const auto p1 = load_tmp(trow, -off1);
                    sum = hn::Add(sum, hn::Mul(vtap[k],
                        hn::Add(constrain(hn::Sub(p0, px), vpri, pri_shift),
                                constrain(hn::Sub(p1, px), vpri, pri_shift))));
                    if constexpr (kClamp) {
                        const int off2 = cdef_dirs[4][k]; // dir + 2
                        const int off3 = cdef_dirs[0][k]; // dir - 2
                        const auto s0 = load_tmp(trow, off2);
                        const auto s1 = load_tmp(trow, -off2);
                        const auto s2 = load_tmp(trow, off3);
                        const auto s3 = load_tmp(trow, -off3);
                        const auto vsectap = hn::Set(d, 2 - k);
                        sum = hn::Add(sum, hn::Mul(vsectap,
                            hn::Add(hn::Add(constrain(hn::Sub(s0, px), vsec, sec_shift),
                                            constrain(hn::Sub(s1, px), vsec, sec_shift)),
                                    hn::Add(constrain(hn::Sub(s2, px), vsec, sec_shift),
                                            constrain(hn::Sub(s3, px), vsec, sec_shift)))));
                        minmax(p0, p1, s0, s1, s2, s3);
                    }
                }
                const auto res = hn::Add(px, round(sum));
                if constexpr (kClamp) {
                    store_clamped(drow, res, vmin, vmax);
                } else {
                    store_unclamped(drow, res);
                }
            }
        };
        if (sec_strength) {
            row_loop(std::true_type{});
        } else {
            row_loop(std::false_type{});
        }
    } else { // sec_strength only
        const int sec_shift = damping - hwy_ulog2(sec_strength);
        const auto vsec = hn::BitCast(du16, hn::Set(d, sec_strength));
        Pixel *drow = dst;
        const int16_t *trow = tmp;
        for (int y = 0; y < H; y++, drow += dst_pxstride, trow += tmp_stride) {
            const auto px = load_px(drow);
            auto sum = hn::Zero(d);
            for (int k = 0; k < 2; k++) {
                const int off1 = cdef_dirs[4][k]; // dir + 2
                const int off2 = cdef_dirs[0][k]; // dir - 2
                const auto s0 = load_tmp(trow, off1);
                const auto s1 = load_tmp(trow, -off1);
                const auto s2 = load_tmp(trow, off2);
                const auto s3 = load_tmp(trow, -off2);
                const auto vsectap = hn::Set(d, 2 - k);
                sum = hn::Add(sum, hn::Mul(vsectap,
                    hn::Add(hn::Add(constrain(hn::Sub(s0, px), vsec, sec_shift),
                                    constrain(hn::Sub(s1, px), vsec, sec_shift)),
                            hn::Add(constrain(hn::Sub(s2, px), vsec, sec_shift),
                                    constrain(hn::Sub(s3, px), vsec, sec_shift)))));
            }
            store_unclamped(drow, hn::Add(px, round(sum)));
        }
    }
}

#define CDEF_FILTER_FN(w, h, bpc, sfx) \
void cdef_filter_##w##x##h##_##sfx(uint##bpc##_t *dst, const ptrdiff_t stride, \
                                        const uint##bpc##_t (*left)[2], \
                                        const uint##bpc##_t *const top, \
                                        const uint##bpc##_t *const bottom, \
                                        const int pri_strength, const int sec_strength, \
                                        const int dir, const int damping, \
                                        const int edges HIGHBD_SUFFIX(bpc)) \
{ \
    cdef_filter_block<uint##bpc##_t, w, h>(dst, stride, left, top, bottom, \
                                           pri_strength, sec_strength, dir, \
                                           damping, edges, BD_MINUS_8(bpc)); \
}
#define HIGHBD_SUFFIX(bpc)
#define BD_MINUS_8(bpc) 0
CDEF_FILTER_FN(8, 8, 8, 8bpc)
CDEF_FILTER_FN(4, 8, 8, 8bpc)
CDEF_FILTER_FN(4, 4, 8, 8bpc)
#undef HIGHBD_SUFFIX
#undef BD_MINUS_8
#define HIGHBD_SUFFIX(bpc) , const int bitdepth_max
#define BD_MINUS_8(bpc) hwy_ulog2(bitdepth_max) - 7
CDEF_FILTER_FN(8, 8, 16, 16bpc)
CDEF_FILTER_FN(4, 8, 16, 16bpc)
CDEF_FILTER_FN(4, 4, 16, 16bpc)
#undef HIGHBD_SUFFIX
#undef BD_MINUS_8
#undef CDEF_FILTER_FN

// Port of cdef_find_dir_c (src/cdef_tmpl.c). Each direction's line sums
// (bins) reduce to a diagonal accumulation bins[i + k] += row_i[k] over
// (possibly folded/reversed) rows of the 8x8 block of
// x = (pixel >> bitdepth_min_8) - 128, x in [-128, 127]:
//   dir 0: rows as-is;            dir 4: rows reversed (Reverse8);
//   dir 1: column-pair fold;      dir 3: column-pair fold reversed (Reverse4);
//   dir 7: row-pair fold;         dir 5: row-pair fold, reversed row order;
//   dir 2: per-row sums;          dir 6: column sums.

// bins[k] += row_i[k - i], with row_i produced by row(i); bins must be
// zeroed by the caller (16 entries, only R + W - 1 used). RowFn callback:
// RVV vector types are sizeless, so a vector array is impossible.
template <class D, class RowFn>
static inline void cdef_diag_bins(const D d, RowFn&& row, const int R,
                                  const int W, int16_t *const bins)
{
    for (int i = 0; i < R; i++)
        hn::StoreN(hn::Add(hn::LoadN(d, bins + i, W), row(i)), d, bins + i, W);
}

// Cost weights of div_table in cdef_find_dir_c: symmetric pairs for the
// diagonal directions (0 and 4, 15 bins), and the 5 full lines + 3 pairs of
// the odd directions (11 bins).
alignas(16) static const int32_t kCdefWDiag[16] = { 840, 420, 280, 210, 168,
    140, 120, 105, 120, 140, 168, 210, 280, 420, 840, 0 };
alignas(16) static const int32_t kCdefWOdd[16] = { 420, 210, 140, 105, 105,
    105, 105, 105, 140, 210, 420, 0, 0, 0, 0, 0 };
alignas(16) static const int32_t kCdefW105[16] = { 105, 105, 105, 105, 105,
    105, 105, 105, 0, 0, 0, 0, 0, 0, 0, 0 };

// Weighted sum of squared bins; w covers 16 bins (zero-padded). |bin| <=
// 8 * 128, so the i32 arithmetic is exact and the lane order cannot change
// the result.
template <class D32>
static inline unsigned cdef_bin_cost(const D32 d32, const int16_t *const bins,
                                     const int32_t *const w)
{
    const hn::Rebind<int16_t, D32> d16;
    const size_t K = hn::Lanes(d32);
    auto acc = hn::Zero(d32);
    for (size_t k = 0; k < 16; k += K) {
        const auto b = hn::PromoteTo(d32, hn::LoadN(d16, bins + k, K));
        acc = hn::Add(acc, hn::Mul(hn::Mul(b, b), hn::LoadN(d32, w + k, K)));
    }
    return (unsigned) hn::ReduceSum(d32, acc);
}

template <typename Pixel>
static int cdef_find_dir(const Pixel *const img, const ptrdiff_t stride,
                         unsigned *const var, const int bitdepth_min_8)
{
    const hn::ScalableTag<int16_t> d;
    const hn::Rebind<Pixel, decltype(d)> dp;
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);

    // LoadN zero-fills lanes [8, Lanes), which stays inert: every step is
    // per-lane or works on independent 8-lane blocks. The shift is exact in
    // int16 because pixel values are <= 4095 (non-negative).
    alignas(16) int16_t xb[64];
    for (int y = 0; y < 8; y++)
        hn::StoreN(hn::Sub(hn::ShiftRightSame(
                               hwy_widen(d, hn::LoadN(dp, img + y * pxstride, 8)),
                               bitdepth_min_8),
                           hn::Set(d, 128)),
                   d, xb + y * 8, 8);

    int16_t b0[16] = {}, b1[16] = {}, b3[16] = {}, b4[16] = {}, b5[16] = {},
            b7[16] = {};
    int16_t rs[16] = {}, cc[16] = {};

    const auto row = [&](const int i) { return hn::LoadN(d, xb + i * 8, 8); };
    cdef_diag_bins(d, row, 8, 8, b0);
    cdef_diag_bins(d, [&](const int i) { return hn::Reverse8(d, row(i)); },
                   8, 8, b4);
    const auto fold = [&](const int i) {
        const auto v = row(i);
        return hn::Add(hn::ConcatEven(d, v, v), hn::ConcatOdd(d, v, v));
    };
    cdef_diag_bins(d, fold, 8, 4, b1);
    cdef_diag_bins(d, [&](const int i) { return hn::Reverse4(d, fold(i)); },
                   8, 4, b3);
    const auto zrow = [&](const int h) {
        return hn::Add(row(2 * h), row(2 * h + 1));
    };
    cdef_diag_bins(d, zrow, 4, 8, b7);
    cdef_diag_bins(d, [&](const int i) { return zrow(3 - i); }, 4, 8, b5);

    auto colsum = hn::Zero(d);
    for (int i = 0; i < 8; i++) {
        const auto v = row(i);
        colsum = hn::Add(colsum, v);
        rs[i] = (int16_t) hn::ReduceSum(d, v);
    }
    hn::StoreN(colsum, d, cc, 8);

    const hn::Repartition<int32_t, decltype(d)> d32;
    unsigned cost[8];
    cost[0] = cdef_bin_cost(d32, b0, kCdefWDiag);
    cost[1] = cdef_bin_cost(d32, b1, kCdefWOdd);
    cost[2] = cdef_bin_cost(d32, rs, kCdefW105);
    cost[3] = cdef_bin_cost(d32, b3, kCdefWOdd);
    cost[4] = cdef_bin_cost(d32, b4, kCdefWDiag);
    cost[5] = cdef_bin_cost(d32, b5, kCdefWOdd);
    cost[6] = cdef_bin_cost(d32, cc, kCdefW105);
    cost[7] = cdef_bin_cost(d32, b7, kCdefWOdd);

    int best_dir = 0;
    unsigned best_cost = cost[0];
    for (int n = 1; n < 8; n++) {
        if (cost[n] > best_cost) {
            best_cost = cost[n];
            best_dir = n;
        }
    }

    // best_cost is the maximum, so the subtraction is non-negative and the
    // C unsigned shift is exact here as well.
    *var = (best_cost - cost[best_dir ^ 4]) >> 10;
    return best_dir;
}

int cdef_find_dir_8bpc(const uint8_t *const img, const ptrdiff_t stride,
                       unsigned *const var)
{
    return cdef_find_dir(img, stride, var, 0);
}

int cdef_find_dir_16bpc(const uint16_t *const img, const ptrdiff_t stride,
                        unsigned *const var, const int bitdepth_max)
{
    return cdef_find_dir(img, stride, var, hwy_ulog2(bitdepth_max) - 7);
}

}  // namespace HWY_NAMESPACE
}  // namespace dav1d

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace dav1d {
HWY_EXPORT(cdef_filter_8x8_8bpc);
HWY_EXPORT(cdef_filter_4x8_8bpc);
HWY_EXPORT(cdef_filter_4x4_8bpc);
HWY_EXPORT(cdef_filter_8x8_16bpc);
HWY_EXPORT(cdef_filter_4x8_16bpc);
HWY_EXPORT(cdef_filter_4x4_16bpc);
HWY_EXPORT(cdef_find_dir_8bpc);
HWY_EXPORT(cdef_find_dir_16bpc);
}  // namespace dav1d

namespace {
// Mirrors of Dav1dCdefDSPContext (src/cdef.h), so that this file does not
// need dav1d's bitdepth-templated C headers.
using CdefFn8 = void (*)(uint8_t *, ptrdiff_t, const uint8_t (*)[2],
                         const uint8_t *, const uint8_t *,
                         int, int, int, int, int);
using CdefDirFn8 = int (*)(const uint8_t *, ptrdiff_t, unsigned *);
struct CdefDSP8 {
    CdefDirFn8 dir;
    CdefFn8 fb[3];
};

using CdefFn16 = void (*)(uint16_t *, ptrdiff_t, const uint16_t (*)[2],
                          const uint16_t *, const uint16_t *,
                          int, int, int, int, int, int);
using CdefDirFn16 = int (*)(const uint16_t *, ptrdiff_t, unsigned *, int);
struct CdefDSP16 {
    CdefDirFn16 dir;
    CdefFn16 fb[3];
};
}  // namespace

namespace dav1d {

static void cdef_dsp_init_8bpc_hwy(void *const c) {
    auto *const ctx = static_cast<CdefDSP8 *>(c);
    ctx->dir = HWY_DYNAMIC_POINTER(cdef_find_dir_8bpc);
    ctx->fb[0] = HWY_DYNAMIC_POINTER(cdef_filter_8x8_8bpc);
    ctx->fb[1] = HWY_DYNAMIC_POINTER(cdef_filter_4x8_8bpc);
    ctx->fb[2] = HWY_DYNAMIC_POINTER(cdef_filter_4x4_8bpc);
}

static void cdef_dsp_init_16bpc_hwy(void *const c) {
    auto *const ctx = static_cast<CdefDSP16 *>(c);
    ctx->dir = HWY_DYNAMIC_POINTER(cdef_find_dir_16bpc);
    ctx->fb[0] = HWY_DYNAMIC_POINTER(cdef_filter_8x8_16bpc);
    ctx->fb[1] = HWY_DYNAMIC_POINTER(cdef_filter_4x8_16bpc);
    ctx->fb[2] = HWY_DYNAMIC_POINTER(cdef_filter_4x4_16bpc);
}

}  // namespace dav1d

extern "C" void dav1d_cdef_dsp_init_hwy_8bpc(void *const c) {
    dav1d::hwy_init_chosen_target();
    dav1d::cdef_dsp_init_8bpc_hwy(c);
}

extern "C" void dav1d_cdef_dsp_init_hwy_16bpc(void *const c) {
    dav1d::hwy_init_chosen_target();
    dav1d::cdef_dsp_init_16bpc_hwy(c);
}

#endif  // HWY_ONCE
