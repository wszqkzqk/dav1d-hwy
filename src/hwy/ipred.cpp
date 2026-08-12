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
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Intra predictors (src/ipred_tmpl.c) implemented with Google Highway: one
// source is compiled per SIMD target and the best one supported by the CPU
// is selected at runtime (HWY_DYNAMIC_DISPATCH). Bit-exact with the C code;
// the cfl_ac/cfl_pred functions and 16bpc smooth_v are not covered.
//
// Compute vectors are 8 x int16 (128 bits); 32-bit math is done on
// sequential half vectors (PromoteTo of Lower/UpperHalf), the same idiom as
// src/hwy/mc_compound.cpp.

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/hwy/ipred.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "src/hwy/common.h"

// Defined in src/tables.c.
extern "C" const uint8_t dav1d_sm_weights[128];
extern "C" const uint16_t dav1d_dr_intra_derivative[44];
extern "C" const int8_t dav1d_filter_intra_taps[5][64];

HWY_BEFORE_NAMESPACE();

namespace dav1d {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// iclip matches include/common/intops.h.
static inline int hwy_iclip(const int v, const int min, const int max) {
    return v < min ? min : v > max ? max : v;
}

// Pixels never exceed 4095, so the u16 -> i16 bitcasts and the i16 -> pixel
// narrowing below are exact for in-range values.
template <class D16, typename Pixel>
static HWY_INLINE hn::VFromD<D16> LoadPx(const D16 d, const Pixel *const p) {
    if constexpr (sizeof(Pixel) == 1) {
        const hn::Rebind<uint8_t, D16> d8;
        const hn::Rebind<uint16_t, D16> du16;
        return hn::BitCast(d, hn::PromoteTo(du16, hn::LoadU(d8, p)));
    } else {
        const hn::Rebind<uint16_t, D16> du16;
        return hn::BitCast(d, hn::LoadU(du16, p));
    }
}

template <class D16, typename Pixel>
static HWY_INLINE hn::VFromD<D16> LoadPxN(const D16 d, const Pixel *const p,
                                          const int n) {
    if constexpr (sizeof(Pixel) == 1) {
        const hn::Rebind<uint8_t, D16> d8;
        const hn::Rebind<uint16_t, D16> du16;
        return hn::BitCast(d, hn::PromoteTo(du16, hn::LoadN(d8, p, n)));
    } else {
        const hn::Rebind<uint16_t, D16> du16;
        return hn::BitCast(d, hn::LoadN(du16, p, n));
    }
}

// v must already be in [0, bitdepth_max].
template <class D16, typename Pixel>
static HWY_INLINE void StorePx(const D16, Pixel *const p,
                               const hn::VFromD<D16> v) {
    const hn::Rebind<Pixel, D16> dp;
    if constexpr (sizeof(Pixel) == 1) {
        hn::StoreU(hn::DemoteTo(dp, v), dp, p);
    } else {
        hn::StoreU(hn::BitCast(dp, v), dp, p);
    }
}

template <class D16, typename Pixel>
static HWY_INLINE void StorePxN(const D16, Pixel *const p,
                                const hn::VFromD<D16> v, const int n) {
    const hn::Rebind<Pixel, D16> dp;
    if constexpr (sizeof(Pixel) == 1) {
        hn::StoreN(hn::DemoteTo(dp, v), dp, p, n);
    } else {
        hn::StoreN(hn::BitCast(dp, v), dp, p, n);
    }
}

// Zero-extending variants for i16 lanes holding u16 values (the smooth
// predictors' partial sums reach 65280).
template <class D32, class V16>
static HWY_INLINE hn::VFromD<D32> WidenLoU(const D32 d32, const V16 v) {
    const hn::RebindToUnsigned<hn::DFromV<V16>> du;
    return hn::PromoteTo(d32, hn::LowerHalf(hn::BitCast(du, v)));
}
template <class D32, class V16>
static HWY_INLINE hn::VFromD<D32> WidenHiU(const D32 d32, const V16 v) {
    const hn::RebindToUnsigned<hn::DFromV<V16>> du;
    return hn::PromoteUpperTo(d32, hn::BitCast(du, v));
}

template <typename Pixel>
static inline void hwy_pixel_set(Pixel *const dst, const Pixel v, const int n) {
    const hn::ScalableTag<Pixel> d;
    const int N = (int) hn::Lanes(d);
    const auto vv = hn::Set(d, v);
    int x = 0;
    for (; x + N <= n; x += N) hn::StoreU(vv, d, dst + x);
    if (x < n) hn::StoreN(vv, d, dst + x, n - x);
}

// Scalar edge-preparation helpers ported from src/ipred_tmpl.c.

static int hwy_get_filter_strength(const int wh, const int angle,
                                   const int is_sm) {
    if (is_sm) {
        if (wh <= 8) {
            if (angle >= 64) return 2;
            if (angle >= 40) return 1;
        } else if (wh <= 16) {
            if (angle >= 48) return 2;
            if (angle >= 20) return 1;
        } else if (wh <= 24) {
            if (angle >=  4) return 3;
        } else {
            return 3;
        }
    } else {
        if (wh <= 8) {
            if (angle >= 56) return 1;
        } else if (wh <= 16) {
            if (angle >= 40) return 1;
        } else if (wh <= 24) {
            if (angle >= 32) return 3;
            if (angle >= 16) return 2;
            if (angle >=  8) return 1;
        } else if (wh <= 32) {
            if (angle >= 32) return 3;
            if (angle >=  4) return 2;
            return 1;
        } else {
            return 3;
        }
    }
    return 0;
}

template <typename Pixel>
static void hwy_filter_edge(Pixel *const out, const int sz,
                            const int lim_from, const int lim_to,
                            const Pixel *const in, const int from,
                            const int to, const int strength)
{
    static const uint8_t kernel[3][5] = {
        { 0, 4, 8, 4, 0 },
        { 0, 5, 6, 5, 0 },
        { 2, 4, 4, 4, 2 }
    };

    int i = 0;
    for (; i < hwy_imin(sz, lim_from); i++)
        out[i] = in[hwy_iclip(i, from, to - 1)];
    for (; i < hwy_imin(lim_to, sz); i++) {
        int s = 0;
        for (int j = 0; j < 5; j++)
            s += in[hwy_iclip(i - 2 + j, from, to - 1)] * kernel[strength - 1][j];
        out[i] = (Pixel) ((s + 8) >> 4);
    }
    for (; i < sz; i++)
        out[i] = in[hwy_iclip(i, from, to - 1)];
}

static inline int hwy_get_upsample(const int wh, const int angle,
                                   const int is_sm) {
    return angle < 40 && wh <= 16 >> is_sm;
}

template <typename Pixel>
static void hwy_upsample_edge(Pixel *const out, const int hsz,
                              const Pixel *const in, const int from,
                              const int to, const int bitdepth_max)
{
    static const int8_t kernel[4] = { -1, 9, 9, -1 };
    int i;
    for (i = 0; i < hsz - 1; i++) {
        out[i * 2] = in[hwy_iclip(i, from, to - 1)];

        int s = 0;
        for (int j = 0; j < 4; j++)
            s += in[hwy_iclip(i + j - 1, from, to - 1)] * kernel[j];
        out[i * 2 + 1] = (Pixel) hwy_iclip((s + 8) >> 4, 0, bitdepth_max);
    }
    out[i * 2] = in[hwy_iclip(i, from, to - 1)];
}

// Weighted edge interpolation (a0*(64-frac) + a1*frac + 32) >> 6. 8bpc fits
// in 16 bits: both taps are <= 255 and the weights sum to 64, so the sum is
// <= 16320. 16bpc needs the i32 halves (4095*64); the result is always in
// [0, bitdepth_max], so the narrowing packs/stores are exact.
template <typename Pixel, class D16, class D32>
static HWY_INLINE hn::VFromD<D16> z_interp(const D16 d, const D32 d32,
                                           const hn::VFromD<D16> a0,
                                           const hn::VFromD<D16> a1,
                                           const hn::VFromD<D16> vf) {
    if constexpr (sizeof(Pixel) == 1) {
        const auto v = hn::Add(hn::Mul(a0, hn::Sub(hn::Set(d, 64), vf)),
                               hn::Mul(a1, vf));
        return hn::ShiftRight<6>(hn::Add(v, hn::Set(d, 32)));
    } else {
        const auto v64 = hn::Set(d32, 64);
        const auto lo = hn::Add(hn::Mul(WidenLo(d32, a0),
                                        hn::Sub(v64, WidenLo(d32, vf))),
                                hn::Mul(WidenLo(d32, a1), WidenLo(d32, vf)));
        const auto hi = hn::Add(hn::Mul(WidenHi(d32, a0),
                                        hn::Sub(v64, WidenHi(d32, vf))),
                                hn::Mul(WidenHi(d32, a1), WidenHi(d32, vf)));
        const auto v32 = hn::Set(d32, 32);
        return PackHalves(d, hn::ShiftRight<6>(hn::Add(lo, v32)),
                          hn::ShiftRight<6>(hn::Add(hi, v32)));
    }
}

// Rows of the z1/z2 directional predictors: dst[x] = interp(top[b], top[b+1])
// with b = base0 + (x - x0) * inc for x in [x0, x1); inc is 2 for upsampled
// edges, else 1. Loads never exceed the index range the C code reads: a full
// chunk's highest tap index is the one its last lane uses (x1 excludes lanes
// the C code would not compute).
template <typename Pixel>
static void z_interp_row(Pixel *const dst, const Pixel *const top,
                         const int x0, const int x1, const int base0,
                         const int inc, const int frac)
{
    const hn::CappedTag<int16_t, 8> d;
    const hn::Repartition<int32_t, decltype(d)> d32;
    const auto vf = hn::Set(d, frac);
    int x = x0, b = base0;
    if (inc == 1) {
        for (; x + 8 <= x1; x += 8, b += 8) {
            const auto a0 = LoadPx(d, top + b);
            const auto a1 = LoadPx(d, top + b + 1);
            StorePx(d, dst + x, z_interp<Pixel>(d, d32, a0, a1, vf));
        }
    } else if constexpr (sizeof(Pixel) == 1) {
        // Upsampled edge: adjacent tap pairs are the even/odd bytes of a
        // 16-byte window; the u16 bitcast + mask/shift widens and
        // deinterleaves in one step.
        const hn::CappedTag<uint8_t, 16> d8;
        const hn::Repartition<uint16_t, decltype(d8)> du16;
        for (; x + 8 <= x1; x += 8, b += 16) {
            const auto v = hn::BitCast(du16, hn::LoadU(d8, top + b));
            const auto a0 = hn::BitCast(d, hn::And(v, hn::Set(du16, 0xff)));
            const auto a1 = hn::BitCast(d, hn::ShiftRight<8>(v));
            StorePx(d, dst + x, z_interp<Pixel>(d, d32, a0, a1, vf));
        }
    } else {
        const hn::Rebind<uint16_t, decltype(d)> du16;
        for (; x + 8 <= x1; x += 8, b += 16) {
            const auto v0 = hn::LoadU(du16, top + b);
            const auto v1 = hn::LoadU(du16, top + b + 8);
            const auto a0 = hn::BitCast(d, hn::ConcatEven(du16, v1, v0));
            const auto a1 = hn::BitCast(d, hn::ConcatOdd(du16, v1, v0));
            StorePx(d, dst + x, z_interp<Pixel>(d, d32, a0, a1, vf));
        }
    }
    for (; x < x1; x++, b += inc)
        dst[x] = (Pixel) ((top[b] * (64 - frac) + top[b + 1] * frac + 32) >> 6);
}

template <typename Pixel>
static void ipred_z1_hwy(Pixel *dst, const ptrdiff_t stride,
                         const Pixel *const topleft_in,
                         const int width, const int height, int angle,
                         const int bitdepth_max)
{
    const int is_sm = (angle >> 9) & 0x1;
    const int enable_intra_edge_filter = angle >> 10;
    angle &= 511;
    int dx = (int) dav1d_dr_intra_derivative[angle >> 1];
    Pixel top_out[64 + 64];
    const Pixel *top;
    int max_base_x;
    const int upsample_above = enable_intra_edge_filter ?
        hwy_get_upsample(width + height, 90 - angle, is_sm) : 0;
    if (upsample_above) {
        hwy_upsample_edge(top_out, width + height, &topleft_in[1], -1,
                          width + hwy_imin(width, height), bitdepth_max);
        top = top_out;
        max_base_x = 2 * (width + height) - 2;
        dx <<= 1;
    } else {
        const int filter_strength = enable_intra_edge_filter ?
            hwy_get_filter_strength(width + height, 90 - angle, is_sm) : 0;
        if (filter_strength) {
            hwy_filter_edge(top_out, width + height, 0, width + height,
                            &topleft_in[1], -1, width + hwy_imin(width, height),
                            filter_strength);
            top = top_out;
            max_base_x = width + height - 1;
        } else {
            top = &topleft_in[1];
            max_base_x = width + hwy_imin(width, height) - 1;
        }
    }
    const int base_inc = 1 + upsample_above;
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);
    for (int y = 0, xpos = dx; y < height;
         y++, xpos += dx, dst += pxstride)
    {
        const int frac = xpos & 0x3E;
        const int base0 = xpos >> 6;
        // First x whose base >= max_base_x; the C code fills the rest of the
        // row with top[max_base_x].
        const int rem = max_base_x - base0;
        int x_stop = width;
        if (rem <= 0) {
            x_stop = 0;
        } else if (rem < width * base_inc) {
            x_stop = (rem + base_inc - 1) / base_inc;
        }
        z_interp_row<Pixel>(dst, top, 0, x_stop, base0, base_inc, frac);
        if (x_stop < width)
            hwy_pixel_set(dst + x_stop, top[max_base_x], width - x_stop);
    }
}

template <typename Pixel>
static void ipred_z2_hwy(Pixel *dst, const ptrdiff_t stride,
                         const Pixel *const topleft_in,
                         const int width, const int height, int angle,
                         const int max_width, const int max_height,
                         const int bitdepth_max)
{
    const int is_sm = (angle >> 9) & 0x1;
    const int enable_intra_edge_filter = angle >> 10;
    angle &= 511;
    int dy = (int) dav1d_dr_intra_derivative[(angle - 90) >> 1];
    int dx = (int) dav1d_dr_intra_derivative[(180 - angle) >> 1];
    const int upsample_left = enable_intra_edge_filter ?
        hwy_get_upsample(width + height, 180 - angle, is_sm) : 0;
    const int upsample_above = enable_intra_edge_filter ?
        hwy_get_upsample(width + height, angle - 90, is_sm) : 0;
    Pixel edge[64 + 64 + 1];
    Pixel *const topleft = &edge[64];

    if (upsample_above) {
        hwy_upsample_edge(topleft, width + 1, topleft_in, 0, width + 1,
                          bitdepth_max);
        dx <<= 1;
    } else {
        const int filter_strength = enable_intra_edge_filter ?
            hwy_get_filter_strength(width + height, angle - 90, is_sm) : 0;
        if (filter_strength) {
            hwy_filter_edge(&topleft[1], width, 0, max_width,
                            &topleft_in[1], -1, width, filter_strength);
        } else {
            memcpy(&topleft[1], &topleft_in[1], width * sizeof(Pixel));
        }
    }
    if (upsample_left) {
        hwy_upsample_edge(&topleft[-height * 2], height + 1,
                          &topleft_in[-height], 0, height + 1, bitdepth_max);
        dy <<= 1;
    } else {
        const int filter_strength = enable_intra_edge_filter ?
            hwy_get_filter_strength(width + height, 180 - angle, is_sm) : 0;
        if (filter_strength) {
            hwy_filter_edge(&topleft[-height], height, height - max_height,
                            height, &topleft_in[-height], 0, height + 1,
                            filter_strength);
        } else {
            memcpy(&topleft[-height], &topleft_in[-height],
                   height * sizeof(Pixel));
        }
    }
    *topleft = *topleft_in;

    const int base_inc_x = 1 + upsample_above;
    const Pixel *const left = &topleft[-(1 + upsample_left)];
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);
    for (int y = 0, xpos = ((1 + upsample_above) << 6) - dx; y < height;
         y++, xpos -= dx, dst += pxstride)
    {
        const int base0 = xpos >> 6;
        const int frac_x = xpos & 0x3E;
        // base_x = base0 + x * base_inc_x grows monotonically, so the row
        // splits at x0: left-edge interpolation below, top-edge at/above.
        int x0 = 0;
        if (base0 < 0) {
            x0 = (-base0 + base_inc_x - 1) / base_inc_x;
            if (x0 > width) x0 = width;
        }
        for (int x = 0, ypos = (y << (6 + upsample_left)) - dy; x < x0;
             x++, ypos -= dy)
        {
            const int base_y = ypos >> 6;
            const int frac_y = ypos & 0x3E;
            const int v = left[-base_y] * (64 - frac_y) +
                          left[-(base_y + 1)] * frac_y;
            dst[x] = (Pixel) ((v + 32) >> 6);
        }
        z_interp_row<Pixel>(dst, topleft, x0, width, base0 + x0 * base_inc_x,
                            base_inc_x, frac_x);
    }
}

template <typename Pixel>
static void ipred_z3_hwy(Pixel *dst, const ptrdiff_t stride,
                         const Pixel *const topleft_in,
                         const int width, const int height, int angle,
                         const int bitdepth_max)
{
    const int is_sm = (angle >> 9) & 0x1;
    const int enable_intra_edge_filter = angle >> 10;
    angle &= 511;
    int dy = (int) dav1d_dr_intra_derivative[(270 - angle) >> 1];
    Pixel left_out[64 + 64];
    const Pixel *left;
    int max_base_y;
    const int upsample_left = enable_intra_edge_filter ?
        hwy_get_upsample(width + height, angle - 180, is_sm) : 0;
    if (upsample_left) {
        hwy_upsample_edge(left_out, width + height,
                          &topleft_in[-(width + height)],
                          hwy_imax(width - height, 0), width + height + 1,
                          bitdepth_max);
        left = &left_out[2 * (width + height) - 2];
        max_base_y = 2 * (width + height) - 2;
        dy <<= 1;
    } else {
        const int filter_strength = enable_intra_edge_filter ?
            hwy_get_filter_strength(width + height, angle - 180, is_sm) : 0;
        if (filter_strength) {
            hwy_filter_edge(left_out, width + height, 0, width + height,
                            &topleft_in[-(width + height)],
                            hwy_imax(width - height, 0), width + height + 1,
                            filter_strength);
            left = &left_out[width + height - 1];
            max_base_y = width + height - 1;
        } else {
            left = &topleft_in[-1];
            max_base_y = height + hwy_imin(width, height) - 1;
        }
    }
    const int base_inc = 1 + upsample_left;
    // Gather table with non-negative indices: rev[i] = left[-i] covers every
    // tap the C loop reads (max_base_y <= w+h-1 <= 127, or <= 2*(w+h)-2 <= 62
    // when upsampled, which requires w+h <= 16).
    Pixel rev[128];
    for (int i = 0; i <= max_base_y; i++) rev[i] = left[-i];
    const hn::CappedTag<int16_t, 8> d;
    const hn::Repartition<int32_t, decltype(d)> d32;
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);
    if constexpr (sizeof(Pixel) == 1) {
        // Per-column constants of the C code's (x, ypos) walk. The base is
        // pre-clamped to the table size; the fill test is unaffected because
        // max_base_y <= 127: min(c, 127) + r >= mby iff c + r >= mby.
        uint8_t colbase[64] = { 0 };
        uint8_t colfrac[64] = { 0 };
        for (int x = 0, ypos = dy; x < width; x++, ypos += dy) {
            colbase[x] = (uint8_t) hwy_imin(ypos >> 6, 127);
            colfrac[x] = (uint8_t) (ypos & 0x3E);
        }
        const hn::CappedTag<uint8_t, 16> d8;
        const hn::Half<decltype(d8)> d8h;
        const hn::Rebind<uint16_t, decltype(d8h)> du16;
        const int nseg = (max_base_y >> 4) + 1;
        const auto vmby = hn::Set(d8, (uint8_t) max_base_y);
        const auto vfill = hn::Set(d8, rev[max_base_y]);
        const auto v127 = hn::Set(d8, 127);
        const auto v1 = hn::Set(d8, 1);
        const auto v16 = hn::Set(d8, 16);
        const auto v64 = hn::Set(d, 64);
        const auto v32 = hn::Set(d, 32);
        for (int y = 0; y < height; y++, dst += pxstride) {
            // y * base_inc <= 63 * 2 and colbase <= 127, so the u8 add below
            // cannot wrap.
            const auto voff = hn::Set(d8, (uint8_t) (y * base_inc));
            for (int x = 0; x < width; x += 16) {
                const int n = hwy_imin(16, width - x);
                const auto sum = hn::Add(hn::LoadU(d8, colbase + x), voff);
                // Lanes at/past max_base_y take the fill value.
                const auto fill = hn::Ge(sum, vmby);
                const auto idx0 = hn::Min(sum, v127);
                const auto idx1 = hn::Add(idx0, v1);
                // 16-byte segment gathers; Or0 zeroes out-of-segment
                // indices (also for the wrapped idx-16s values >= 240).
                auto a0 = hn::Zero(d8);
                auto a1 = hn::Zero(d8);
                auto s0 = idx0;
                auto s1 = idx1;
                for (int s = 0; s < nseg; s++) {
                    const auto seg = hn::LoadU(d8, rev + 16 * s);
                    a0 = hn::Or(a0, hn::TableLookupBytesOr0(seg, s0));
                    a1 = hn::Or(a1, hn::TableLookupBytesOr0(seg, s1));
                    s0 = hn::Sub(s0, v16);
                    s1 = hn::Sub(s1, v16);
                }
                const auto vf8 = hn::LoadU(d8, colfrac + x);
                // Interpolation in 16 bits on sequential halves; both taps
                // are <= 255 and the weights sum to 64, so the sum fits.
                const auto half = [&](const hn::VFromD<decltype(d8h)> f8,
                                      const hn::VFromD<decltype(d8h)> b0,
                                      const hn::VFromD<decltype(d8h)> b1) {
                    const auto vf = hn::BitCast(d, hn::PromoteTo(du16, f8));
                    const auto t0 = hn::BitCast(d, hn::PromoteTo(du16, b0));
                    const auto t1 = hn::BitCast(d, hn::PromoteTo(du16, b1));
                    const auto v = hn::Add(hn::Mul(t0, hn::Sub(v64, vf)),
                                           hn::Mul(t1, vf));
                    return hn::ShiftRight<6>(hn::Add(v, v32));
                };
                const auto lo = half(hn::LowerHalf(d8h, vf8),
                                     hn::LowerHalf(d8h, a0),
                                     hn::LowerHalf(d8h, a1));
                const auto hi = half(hn::UpperHalf(d8h, vf8),
                                     hn::UpperHalf(d8h, a0),
                                     hn::UpperHalf(d8h, a1));
                const auto res = hn::IfThenElse(fill, vfill,
                    hn::Combine(d8, hn::DemoteTo(d8h, hi),
                                hn::DemoteTo(d8h, lo)));
                if (n == 16) {
                    hn::StoreU(res, d8, dst + x);
                } else {
                    hn::StoreN(res, d8, dst + x, n);
                }
            }
        }
    } else {
        // Per-column constants of the C code's (x, ypos) walk.
        int colbase[64] = { 0 };
        uint8_t colfrac[64] = { 0 };
        for (int x = 0, ypos = dy; x < width; x++, ypos += dy) {
            colbase[x] = ypos >> 6;
            colfrac[x] = (uint8_t) (ypos & 0x3E);
        }
        const hn::Rebind<uint8_t, decltype(d)> d8;
        const hn::Rebind<uint16_t, decltype(d)> du16;
        Pixel a0buf[8], a1buf[8];
        for (int i = 0; i < 8; i++) a0buf[i] = a1buf[i] = 0;
        for (int y = 0; y < height; y++, dst += pxstride) {
            const int rowoff = y * base_inc;
            for (int x = 0; x < width; x += 8) {
                const int n = hwy_imin(8, width - x);
                // Lanes with base >= max_base_y take the fill value; feeding
                // it as both taps makes the interpolation an exact identity.
                for (int k = 0; k < n; k++) {
                    const int b = colbase[x + k] + rowoff;
                    if (b >= max_base_y) {
                        a0buf[k] = a1buf[k] = rev[max_base_y];
                    } else {
                        a0buf[k] = rev[b];
                        a1buf[k] = rev[b + 1];
                    }
                }
                const auto a0 = LoadPx(d, a0buf);
                const auto a1 = LoadPx(d, a1buf);
                const auto vf = hn::BitCast(d, hn::PromoteTo(du16,
                    hn::LoadN(d8, colfrac + x, n)));
                StorePxN(d, dst + x, z_interp<Pixel>(d, d32, a0, a1, vf), n);
            }
        }
    }
}

template <typename Pixel>
static unsigned hwy_dc_gen_top(const Pixel *const topleft, const int width) {
    unsigned dc = width >> 1;
    for (int i = 0; i < width; i++)
        dc += topleft[1 + i];
    return dc >> hwy_ctz(width);
}

template <typename Pixel>
static unsigned hwy_dc_gen_left(const Pixel *const topleft, const int height) {
    unsigned dc = height >> 1;
    // Sums in forward memory order; exact (integer addition is commutative).
    for (int i = 0; i < height; i++)
        dc += topleft[i - height];
    return dc >> hwy_ctz(height);
}

template <typename Pixel>
static unsigned hwy_dc_gen(const Pixel *const topleft,
                           const int width, const int height)
{
    constexpr unsigned kMult1x2 = sizeof(Pixel) == 1 ? 0x5556 : 0xAAAB;
    constexpr unsigned kMult1x4 = sizeof(Pixel) == 1 ? 0x3334 : 0x6667;
    constexpr int kBaseShift = sizeof(Pixel) == 1 ? 16 : 17;
    unsigned dc = (width + height) >> 1;
    for (int i = 0; i < width; i++)
        dc += topleft[i + 1];
    // Forward memory order (addition is commutative), see hwy_dc_gen_left.
    for (int i = 0; i < height; i++)
        dc += topleft[i - height];
    dc >>= hwy_ctz(width + height);

    if (width != height) {
        dc *= (width > height * 2 || height > width * 2) ? kMult1x4 : kMult1x2;
        dc >>= kBaseShift;
    }
    return dc;
}

template <typename Pixel>
static void hwy_splat_dc(Pixel *dst, const ptrdiff_t stride,
                         const int width, const int height, const unsigned dc)
{
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);
    if (width * (int) sizeof(Pixel) >= 64) {
        // Block widths are powers of two, so no tail handling is needed.
        const uint64_t dcN = (uint64_t) dc *
            (sizeof(Pixel) == 1 ? 0x0101010101010101ULL : 0x0001000100010001ULL);
        constexpr int kPerStore = (int) (sizeof(uint64_t) / sizeof(Pixel));
        for (int y = 0; y < height; y++, dst += pxstride)
            for (int x = 0; x < width; x += kPerStore)
                memcpy(dst + x, &dcN, sizeof(dcN));
        return;
    }
    const hn::ScalableTag<Pixel> d;
    const int N = (int) hn::Lanes(d);
    const auto v = hn::Set(d, (Pixel) dc);
    if (width >= N) {
        for (int y = 0; y < height; y++, dst += pxstride)
            for (int x = 0; x < width; x += N) hn::StoreU(v, d, dst + x);
    } else {
        for (int y = 0; y < height; y++, dst += pxstride)
            hn::StoreN(v, d, dst, width);
    }
}

// Smooth predictors. 8bpc is exact in 16-bit lanes: with weights summing to
// 256, pred = 256*edge + w*(px - edge) is in [0, 65280] (the product is in
// [-65025, 65025]); the low-16-bit multiply and the wrapping add are exact
// mod 2^16 and the true sum fits u16, so a final *logical* shift gives the C
// result. The full smooth sums two such parts (up to 17 bits), so only that
// final add/shift widens to i32 (zero-extending). 16bpc needs i32 throughout
// (256*4095 exceeds u16), in the same factored one-multiply-per-term form.
template <typename Pixel>
static void ipred_smooth_hwy(Pixel *dst, const ptrdiff_t stride,
                             const Pixel *const topleft,
                             const int width, const int height)
{
    const hn::CappedTag<int16_t, 8> d;
    const hn::Repartition<int32_t, decltype(d)> d32;
    const hn::Rebind<uint16_t, decltype(d)> du16;
    const hn::Rebind<uint8_t, decltype(d)> d8;
    const uint8_t *const weights_hor = &dav1d_sm_weights[width];
    const uint8_t *const weights_ver = &dav1d_sm_weights[height];
    const int right = topleft[width], bottom = topleft[-height];
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);

    if constexpr (sizeof(Pixel) == 1) {
        const auto vbot = hn::Set(d, bottom);
        const auto vb256 = hn::Set(d, 256 * bottom);
        const auto vr256 = hn::Set(d, 256 * right);
        const auto v256 = hn::Set(d32, 256);
        for (int y = 0; y < height; y++, dst += pxstride) {
            const auto vwv = hn::Set(d, weights_ver[y]);
            const auto vdiff = hn::Set(d, (int16_t) (topleft[-(1 + y)] - right));
            const auto calc = [&](const hn::VFromD<decltype(d)> t16,
                                  const hn::VFromD<decltype(d)> w16,
                                  const int x, const int n) {
                const auto s1 = hn::Add(hn::Mul(hn::Sub(t16, vbot), vwv), vb256);
                const auto s2 = hn::Add(hn::Mul(w16, vdiff), vr256);
                const auto lo = hn::ShiftRight<9>(hn::Add(
                    hn::Add(WidenLoU(d32, s1), WidenLoU(d32, s2)), v256));
                const auto hi = hn::ShiftRight<9>(hn::Add(
                    hn::Add(WidenHiU(d32, s1), WidenHiU(d32, s2)), v256));
                if (n == 8) {
                    StorePx(d, dst + x, PackHalves(d, lo, hi));
                } else {
                    StorePxN(d, dst + x, PackHalves(d, lo, hi), n);
                }
            };
            int x = 0;
            for (; x + 8 <= width; x += 8)
                calc(LoadPx(d, topleft + 1 + x),
                     hn::BitCast(d, hn::PromoteTo(du16,
                         hn::LoadU(d8, weights_hor + x))), x, 8);
            if (x < width)
                calc(LoadPxN(d, topleft + 1 + x, width - x),
                     hn::BitCast(d, hn::PromoteTo(du16,
                         hn::LoadN(d8, weights_hor + x, width - x))),
                     x, width - x);
        }
    } else {
        const auto vconst = hn::Set(d32, 256 * (bottom + right) + 256);
        const auto vbot = hn::Set(d32, bottom);
        for (int y = 0; y < height; y++, dst += pxstride) {
            const auto vwv = hn::Set(d32, weights_ver[y]);
            const auto vdiff = hn::Set(d32, (int32_t) topleft[-(1 + y)] - right);
            const auto calc = [&](const hn::VFromD<decltype(d32)> t32,
                                  const hn::VFromD<decltype(d32)> w32) {
                const auto s = hn::Add(hn::Mul(vwv, hn::Sub(t32, vbot)),
                                       hn::Mul(w32, vdiff));
                return hn::ShiftRight<9>(hn::Add(s, vconst));
            };
            int x = 0;
            for (; x + 8 <= width; x += 8) {
                const auto t = LoadPx(d, topleft + 1 + x);
                const auto w = hn::BitCast(d, hn::PromoteTo(du16,
                    hn::LoadU(d8, weights_hor + x)));
                StorePx(d, dst + x, PackHalves(d, calc(WidenLo(d32, t),
                        WidenLo(d32, w)), calc(WidenHi(d32, t),
                        WidenHi(d32, w))));
            }
            if (x < width) {
                const int n = width - x;
                const auto t = LoadPxN(d, topleft + 1 + x, n);
                const auto w = hn::BitCast(d, hn::PromoteTo(du16,
                    hn::LoadN(d8, weights_hor + x, n)));
                StorePxN(d, dst + x, PackHalves(d, calc(WidenLo(d32, t),
                         WidenLo(d32, w)), calc(WidenHi(d32, t),
                         WidenHi(d32, w))), n);
            }
        }
    }
}

template <typename Pixel>
static void ipred_smooth_v_hwy(Pixel *dst, const ptrdiff_t stride,
                               const Pixel *const topleft,
                               const int width, const int height)
{
    const hn::CappedTag<int16_t, 8> d;
    const hn::Repartition<int32_t, decltype(d)> d32;
    const hn::Rebind<uint16_t, decltype(d)> du16;
    const uint8_t *const weights_ver = &dav1d_sm_weights[height];
    const int bottom = topleft[-height];
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);

    if constexpr (sizeof(Pixel) == 1) {
        const auto vconst = hn::Set(d, 256 * bottom + 128);
        const auto vbot = hn::Set(d, bottom);
        for (int y = 0; y < height; y++, dst += pxstride) {
            const auto vwv = hn::Set(d, weights_ver[y]);
            const auto calc = [&](const hn::VFromD<decltype(d)> t16) {
                const auto s = hn::Add(hn::Mul(hn::Sub(t16, vbot), vwv), vconst);
                return hn::BitCast(d, hn::ShiftRight<8>(hn::BitCast(du16, s)));
            };
            int x = 0;
            for (; x + 8 <= width; x += 8)
                StorePx(d, dst + x, calc(LoadPx(d, topleft + 1 + x)));
            if (x < width) {
                const int n = width - x;
                StorePxN(d, dst + x, calc(LoadPxN(d, topleft + 1 + x, n)), n);
            }
        }
    } else {
        const auto vconst = hn::Set(d32, 256 * bottom + 128);
        const auto vbot = hn::Set(d32, bottom);
        for (int y = 0; y < height; y++, dst += pxstride) {
            const auto vwv = hn::Set(d32, weights_ver[y]);
            const auto calc = [&](const hn::VFromD<decltype(d32)> t32) {
                const auto s = hn::Add(hn::Mul(vwv, hn::Sub(t32, vbot)), vconst);
                return hn::ShiftRight<8>(s);
            };
            int x = 0;
            for (; x + 8 <= width; x += 8) {
                const auto t = LoadPx(d, topleft + 1 + x);
                StorePx(d, dst + x, PackHalves(d, calc(WidenLo(d32, t)),
                                               calc(WidenHi(d32, t))));
            }
            if (x < width) {
                const int n = width - x;
                const auto t = LoadPxN(d, topleft + 1 + x, n);
                StorePxN(d, dst + x, PackHalves(d, calc(WidenLo(d32, t)),
                                                calc(WidenHi(d32, t))), n);
            }
        }
    }
}

template <typename Pixel>
static void ipred_smooth_h_hwy(Pixel *dst, const ptrdiff_t stride,
                               const Pixel *const topleft,
                               const int width, const int height)
{
    const hn::CappedTag<int16_t, 8> d;
    const hn::Repartition<int32_t, decltype(d)> d32;
    const hn::Rebind<uint16_t, decltype(d)> du16;
    const hn::Rebind<uint8_t, decltype(d)> d8;
    const uint8_t *const weights_hor = &dav1d_sm_weights[width];
    const int right = topleft[width];
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);

    if constexpr (sizeof(Pixel) == 1) {
        const auto vconst = hn::Set(d, 256 * right + 128);
        for (int y = 0; y < height; y++, dst += pxstride) {
            const auto vdiff = hn::Set(d, (int16_t) (topleft[-(y + 1)] - right));
            const auto calc = [&](const hn::VFromD<decltype(d)> w16) {
                const auto s = hn::Add(hn::Mul(w16, vdiff), vconst);
                return hn::BitCast(d, hn::ShiftRight<8>(hn::BitCast(du16, s)));
            };
            int x = 0;
            for (; x + 8 <= width; x += 8)
                StorePx(d, dst + x, calc(hn::BitCast(d, hn::PromoteTo(du16,
                        hn::LoadU(d8, weights_hor + x)))));
            if (x < width) {
                const int n = width - x;
                StorePxN(d, dst + x, calc(hn::BitCast(d, hn::PromoteTo(du16,
                         hn::LoadN(d8, weights_hor + x, n)))), n);
            }
        }
    } else {
        const auto vconst = hn::Set(d32, 256 * right + 128);
        for (int y = 0; y < height; y++, dst += pxstride) {
            const auto vdiff = hn::Set(d32, (int32_t) topleft[-(y + 1)] - right);
            const auto calc = [&](const hn::VFromD<decltype(d32)> w32) {
                return hn::ShiftRight<8>(hn::Add(hn::Mul(w32, vdiff), vconst));
            };
            int x = 0;
            for (; x + 8 <= width; x += 8) {
                const auto w = hn::BitCast(d, hn::PromoteTo(du16,
                    hn::LoadU(d8, weights_hor + x)));
                StorePx(d, dst + x, PackHalves(d, calc(WidenLo(d32, w)),
                                               calc(WidenHi(d32, w))));
            }
            if (x < width) {
                const int n = width - x;
                const auto w = hn::BitCast(d, hn::PromoteTo(du16,
                    hn::LoadN(d8, weights_hor + x, n)));
                StorePxN(d, dst + x, PackHalves(d, calc(WidenLo(d32, w)),
                                                calc(WidenHi(d32, w))), n);
            }
        }
    }
}

// 16-bit lanes hold every intermediate: left + top - topleft is in
// [-4095, 8190] and the absolute differences are <= 8190.
template <typename Pixel>
static void ipred_paeth_hwy(Pixel *dst, const ptrdiff_t stride,
                            const Pixel *const tl_ptr,
                            const int width, const int height)
{
    const hn::CappedTag<int16_t, 8> d;
    const int topleft = tl_ptr[0];
    const auto vtl = hn::Set(d, (int16_t) topleft);
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);

    for (int y = 0; y < height; y++, dst += pxstride) {
        const int left = tl_ptr[-(y + 1)];
        const auto vleft = hn::Set(d, (int16_t) left);
        const auto vptop = hn::Set(d, (int16_t) hwy_iabs(left - topleft));
        // Same tie-breaking order as the C code.
        const auto calc = [&](const hn::VFromD<decltype(d)> top) {
            const auto base = hn::Sub(hn::Add(vleft, top), vtl);
            const auto pleft = hn::Abs(hn::Sub(top, vtl));
            const auto ptl = hn::Abs(hn::Sub(base, vtl));
            const auto use_left = hn::And(hn::Le(pleft, vptop),
                                          hn::Le(pleft, ptl));
            return hn::IfThenElse(use_left, vleft,
                                  hn::IfThenElse(hn::Le(vptop, ptl), top, vtl));
        };
        int x = 0;
        for (; x + 8 <= width; x += 8) {
            const auto top = LoadPx(d, tl_ptr + 1 + x);
            StorePx(d, dst + x, calc(top));
        }
        if (x < width) {
            const int n = width - x;
            const auto top = LoadPxN(d, tl_ptr + 1 + x, n);
            StorePxN(d, dst + x, calc(top), n);
        }
    }
}

// Port of ipred_filter_c(); scalar: later 4x2 sub-blocks read back the
// pixels predicted for earlier ones. Up to 32x32 only.
template <typename Pixel>
static void ipred_filter_hwy(Pixel *dst, const ptrdiff_t stride,
                             const Pixel *const topleft_in,
                             const int width, const int height, int filt_idx,
                             const int bitdepth_max)
{
    filt_idx &= 511;
    const int8_t *const filter = dav1d_filter_intra_taps[filt_idx];
    const Pixel *top = &topleft_in[1];
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);
    for (int y = 0; y < height; y += 2) {
        const Pixel *topleft = &topleft_in[-y];
        const Pixel *left = &topleft[-1];
        ptrdiff_t left_stride = -1;
        for (int x = 0; x < width; x += 4) {
            const int p0 = *topleft;
            const int p1 = top[0], p2 = top[1], p3 = top[2], p4 = top[3];
            const int p5 = left[0], p6 = left[left_stride];
            Pixel *ptr = &dst[x];
            const int8_t *flt_ptr = filter;

            for (int yy = 0; yy < 2; yy++) {
                for (int xx = 0; xx < 4; xx++, flt_ptr++) {
                    const int acc = flt_ptr[ 0] * p0 + flt_ptr[ 8] * p1 +
                                    flt_ptr[16] * p2 + flt_ptr[24] * p3 +
                                    flt_ptr[32] * p4 + flt_ptr[40] * p5 +
                                    flt_ptr[48] * p6;
                    ptr[xx] = (Pixel) hwy_iclip((acc + 8) >> 4, 0, bitdepth_max);
                }
                ptr += pxstride;
            }
            left = &dst[x + 4 - 1];
            left_stride = pxstride;
            top += 4;
            topleft = &top[-1];
        }
        top = &dst[pxstride];
        dst += pxstride * 2;
    }
}

// The 8-bit path uses a byte table lookup on a 16-byte table (indices are 3
// bits, so the zero padding of the 8-entry palette is never selected); the
// 16-bit path uses a lane table lookup on the 8-entry palette vector. The
// w == 8 path exploits that idx is row-contiguous: one 8-byte chunk covers
// exactly two rows. Other widths stage one row of idx so the 8-byte chunk
// loads never overrun the w*h/2-byte idx plane.
static void pal_pred_8_hwy(uint8_t *dst, const ptrdiff_t stride,
                           const uint8_t *const pal, const uint8_t *idx,
                           const int w, const int h)
{
    const hn::CappedTag<uint8_t, 16> d;
    const hn::Half<decltype(d)> dh;
    const auto tbl = hn::LoadN(d, pal, 8);
    if (w == 8) {
        int y = 0;
        for (; y + 1 < h; y += 2) {
            const auto iv8 = hn::LoadU(dh, idx + 4 * y);
            const auto iv = hn::Combine(d, iv8, iv8);
            const auto lo = hn::TableLookupBytes(tbl, hn::And(iv, hn::Set(d, 7)));
            const auto hi = hn::TableLookupBytes(tbl, hn::ShiftRight<4>(iv));
            const auto out = hn::InterleaveLower(d, lo, hi);
            hn::StoreU(hn::LowerHalf(dh, out), dh, dst);
            hn::StoreU(hn::UpperHalf(dh, out), dh, dst + stride);
            dst += 2 * stride;
        }
        if (y < h) { // h is always even; kept for contract safety
            const uint8_t iv8buf[8] = { idx[4 * y], idx[4 * y + 1],
                                        idx[4 * y + 2], idx[4 * y + 3] };
            const auto iv8 = hn::LoadU(dh, iv8buf);
            const auto iv = hn::Combine(d, iv8, iv8);
            const auto lo = hn::TableLookupBytes(tbl, hn::And(iv, hn::Set(d, 7)));
            const auto hi = hn::TableLookupBytes(tbl, hn::ShiftRight<4>(iv));
            const auto out = hn::InterleaveLower(d, lo, hi);
            hn::StoreU(hn::LowerHalf(dh, out), dh, dst);
        }
        return;
    }
    uint8_t idxbuf[40] = { 0 };
    for (int y = 0; y < h; y++, dst += stride) {
        memcpy(idxbuf, idx, (size_t) (w >> 1));
        idx += w >> 1;
        for (int x = 0; x < w; x += 16) {
            const int n = hwy_imin(16, w - x);
            const auto iv8 = hn::LoadU(dh, idxbuf + (x >> 1));
            const auto iv = hn::Combine(d, iv8, iv8);
            const auto lo = hn::TableLookupBytes(tbl, hn::And(iv, hn::Set(d, 7)));
            const auto hi = hn::TableLookupBytes(tbl, hn::ShiftRight<4>(iv));
            const auto out = hn::InterleaveLower(d, lo, hi);
            if (n == 16) {
                hn::StoreU(out, d, dst + x);
            } else {
                hn::StoreN(out, d, dst + x, n);
            }
        }
    }
}

static void pal_pred_16_hwy(uint16_t *dst, const ptrdiff_t stride,
                            const uint16_t *const pal, const uint8_t *idx,
                            const int w, const int h)
{
    const hn::CappedTag<uint16_t, 8> d16;
    const hn::Rebind<int16_t, decltype(d16)> di16;
    const hn::CappedTag<uint8_t, 8> d8;
    // The 8-entry palette is one vector; indices are 3 bits, in-contract for
    // TableLookupLanes.
    const auto paltbl = hn::BitCast(di16, hn::LoadU(d16, pal));
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(uint16_t);
    const auto gather = [&](const hn::VFromD<decltype(d16)> v) {
        return hn::BitCast(d16, hn::TableLookupLanes(paltbl,
            hn::IndicesFromVec(di16, hn::BitCast(di16, v))));
    };
    if (w == 8) {
        int y = 0;
        for (; y + 1 < h; y += 2) {
            const auto iv = hn::LoadU(d8, idx + 4 * y);
            const auto lo = hn::PromoteTo(d16, hn::And(iv, hn::Set(d8, 7)));
            const auto hi = hn::PromoteTo(d16, hn::ShiftRight<4>(iv));
            const auto glo = gather(lo);
            const auto ghi = gather(hi);
            hn::StoreU(hn::InterleaveLower(d16, glo, ghi), d16, dst);
            hn::StoreU(hn::InterleaveUpper(d16, glo, ghi), d16,
                       dst + pxstride);
            dst += 2 * pxstride;
        }
        if (y < h) { // h is always even; kept for contract safety
            const uint8_t iv8buf[8] = { idx[4 * y], idx[4 * y + 1],
                                        idx[4 * y + 2], idx[4 * y + 3] };
            const auto iv = hn::LoadU(d8, iv8buf);
            const auto lo = hn::PromoteTo(d16, hn::And(iv, hn::Set(d8, 7)));
            const auto hi = hn::PromoteTo(d16, hn::ShiftRight<4>(iv));
            const auto glo = gather(lo);
            const auto ghi = gather(hi);
            hn::StoreU(hn::InterleaveLower(d16, glo, ghi), d16, dst);
        }
        return;
    }
    uint8_t idxbuf[40] = { 0 };
    for (int y = 0; y < h; y++, dst += pxstride) {
        memcpy(idxbuf, idx, (size_t) (w >> 1));
        idx += w >> 1;
        for (int x = 0; x < w; x += 16) {
            const int n = hwy_imin(16, w - x);
            const auto iv = hn::LoadU(d8, idxbuf + (x >> 1));
            const auto lo = hn::PromoteTo(d16, hn::And(iv, hn::Set(d8, 7)));
            const auto hi = hn::PromoteTo(d16, hn::ShiftRight<4>(iv));
            const auto glo = gather(lo);
            const auto ghi = gather(hi);
            const auto out0 = hn::InterleaveLower(d16, glo, ghi);
            if (n < 8) {
                hn::StoreN(out0, d16, dst + x, n);
                continue;
            }
            hn::StoreU(out0, d16, dst + x);
            const auto out1 = hn::InterleaveUpper(d16, glo, ghi);
            if (n == 16) {
                hn::StoreU(out1, d16, dst + x + 8);
            } else {
                hn::StoreN(out1, d16, dst + x + 8, n - 8);
            }
        }
    }
}

#define IPRED_FNS(bpc, sfx) \
void ipred_dc_##sfx(uint##bpc##_t *dst, const ptrdiff_t stride, \
                    const uint##bpc##_t *const topleft, \
                    const int width, const int height, const int a, \
                    const int max_width, const int max_height \
                    HIGHBD_SUFFIX(bpc)) \
{ \
    (void) a; (void) max_width; (void) max_height; (void) BD_MAX(bpc); \
    hwy_splat_dc(dst, stride, width, height, \
                 hwy_dc_gen(topleft, width, height)); \
} \
void ipred_dc_left_##sfx(uint##bpc##_t *dst, const ptrdiff_t stride, \
                         const uint##bpc##_t *const topleft, \
                         const int width, const int height, const int a, \
                         const int max_width, const int max_height \
                         HIGHBD_SUFFIX(bpc)) \
{ \
    (void) a; (void) max_width; (void) max_height; (void) BD_MAX(bpc); \
    (void) width; \
    hwy_splat_dc(dst, stride, width, height, \
                 hwy_dc_gen_left(topleft, height)); \
} \
void ipred_dc_top_##sfx(uint##bpc##_t *dst, const ptrdiff_t stride, \
                        const uint##bpc##_t *const topleft, \
                        const int width, const int height, const int a, \
                        const int max_width, const int max_height \
                        HIGHBD_SUFFIX(bpc)) \
{ \
    (void) a; (void) max_width; (void) max_height; (void) BD_MAX(bpc); \
    (void) height; \
    hwy_splat_dc(dst, stride, width, height, hwy_dc_gen_top(topleft, width)); \
} \
void ipred_dc_128_##sfx(uint##bpc##_t *dst, const ptrdiff_t stride, \
                        const uint##bpc##_t *const topleft, \
                        const int width, const int height, const int a, \
                        const int max_width, const int max_height \
                        HIGHBD_SUFFIX(bpc)) \
{ \
    (void) topleft; (void) a; (void) max_width; (void) max_height; \
    hwy_splat_dc(dst, stride, width, height, (BD_MAX(bpc) + 1) >> 1); \
} \
void ipred_z1_##sfx(uint##bpc##_t *dst, const ptrdiff_t stride, \
                    const uint##bpc##_t *const topleft, \
                    const int width, const int height, const int angle, \
                    const int max_width, const int max_height \
                    HIGHBD_SUFFIX(bpc)) \
{ \
    (void) max_width; (void) max_height; \
    ipred_z1_hwy(dst, stride, topleft, width, height, angle, BD_MAX(bpc)); \
} \
void ipred_z2_##sfx(uint##bpc##_t *dst, const ptrdiff_t stride, \
                    const uint##bpc##_t *const topleft, \
                    const int width, const int height, const int angle, \
                    const int max_width, const int max_height \
                    HIGHBD_SUFFIX(bpc)) \
{ \
    ipred_z2_hwy(dst, stride, topleft, width, height, angle, max_width, \
                 max_height, BD_MAX(bpc)); \
} \
void ipred_z3_##sfx(uint##bpc##_t *dst, const ptrdiff_t stride, \
                    const uint##bpc##_t *const topleft, \
                    const int width, const int height, const int angle, \
                    const int max_width, const int max_height \
                    HIGHBD_SUFFIX(bpc)) \
{ \
    (void) max_width; (void) max_height; \
    ipred_z3_hwy(dst, stride, topleft, width, height, angle, BD_MAX(bpc)); \
} \
void ipred_smooth_##sfx(uint##bpc##_t *dst, const ptrdiff_t stride, \
                        const uint##bpc##_t *const topleft, \
                        const int width, const int height, const int a, \
                        const int max_width, const int max_height \
                        HIGHBD_SUFFIX(bpc)) \
{ \
    (void) a; (void) max_width; (void) max_height; (void) BD_MAX(bpc); \
    ipred_smooth_hwy(dst, stride, topleft, width, height); \
} \
void ipred_smooth_v_##sfx(uint##bpc##_t *dst, const ptrdiff_t stride, \
                          const uint##bpc##_t *const topleft, \
                          const int width, const int height, const int a, \
                          const int max_width, const int max_height \
                          HIGHBD_SUFFIX(bpc)) \
{ \
    (void) a; (void) max_width; (void) max_height; (void) BD_MAX(bpc); \
    ipred_smooth_v_hwy(dst, stride, topleft, width, height); \
} \
void ipred_smooth_h_##sfx(uint##bpc##_t *dst, const ptrdiff_t stride, \
                          const uint##bpc##_t *const topleft, \
                          const int width, const int height, const int a, \
                          const int max_width, const int max_height \
                          HIGHBD_SUFFIX(bpc)) \
{ \
    (void) a; (void) max_width; (void) max_height; (void) BD_MAX(bpc); \
    ipred_smooth_h_hwy(dst, stride, topleft, width, height); \
} \
void ipred_paeth_##sfx(uint##bpc##_t *dst, const ptrdiff_t stride, \
                       const uint##bpc##_t *const topleft, \
                       const int width, const int height, const int a, \
                       const int max_width, const int max_height \
                       HIGHBD_SUFFIX(bpc)) \
{ \
    (void) a; (void) max_width; (void) max_height; (void) BD_MAX(bpc); \
    ipred_paeth_hwy(dst, stride, topleft, width, height); \
} \
void ipred_filter_##sfx(uint##bpc##_t *dst, const ptrdiff_t stride, \
                        const uint##bpc##_t *const topleft, \
                        const int width, const int height, const int filt_idx, \
                        const int max_width, const int max_height \
                        HIGHBD_SUFFIX(bpc)) \
{ \
    (void) max_width; (void) max_height; \
    ipred_filter_hwy(dst, stride, topleft, width, height, filt_idx, \
                     BD_MAX(bpc)); \
} \
void pal_pred_##sfx(uint##bpc##_t *dst, const ptrdiff_t stride, \
                    const uint##bpc##_t *const pal, const uint8_t *idx, \
                    const int w, const int h) \
{ \
    if (bpc == 8) { \
        pal_pred_8_hwy((uint8_t *) dst, stride, (const uint8_t *) pal, idx, \
                       w, h); \
    } else { \
        pal_pred_16_hwy((uint16_t *) dst, stride, (const uint16_t *) pal, \
                        idx, w, h); \
    } \
}

#define HIGHBD_SUFFIX(bpc)
#define BD_MAX(bpc) 255
IPRED_FNS(8, 8bpc)
#undef HIGHBD_SUFFIX
#undef BD_MAX
#define HIGHBD_SUFFIX(bpc) , const int bitdepth_max
#define BD_MAX(bpc) bitdepth_max
IPRED_FNS(16, 16bpc)
#undef HIGHBD_SUFFIX
#undef BD_MAX
#undef IPRED_FNS

}  // namespace HWY_NAMESPACE
}  // namespace dav1d

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace dav1d {
HWY_EXPORT(ipred_dc_8bpc);
HWY_EXPORT(ipred_dc_left_8bpc);
HWY_EXPORT(ipred_dc_top_8bpc);
HWY_EXPORT(ipred_dc_128_8bpc);
HWY_EXPORT(ipred_z1_8bpc);
HWY_EXPORT(ipred_z2_8bpc);
HWY_EXPORT(ipred_z3_8bpc);
HWY_EXPORT(ipred_smooth_8bpc);
HWY_EXPORT(ipred_smooth_v_8bpc);
HWY_EXPORT(ipred_smooth_h_8bpc);
HWY_EXPORT(ipred_paeth_8bpc);
HWY_EXPORT(ipred_filter_8bpc);
HWY_EXPORT(pal_pred_8bpc);
HWY_EXPORT(ipred_dc_16bpc);
HWY_EXPORT(ipred_dc_left_16bpc);
HWY_EXPORT(ipred_dc_top_16bpc);
HWY_EXPORT(ipred_dc_128_16bpc);
HWY_EXPORT(ipred_z1_16bpc);
HWY_EXPORT(ipred_z2_16bpc);
HWY_EXPORT(ipred_z3_16bpc);
HWY_EXPORT(ipred_smooth_16bpc);
HWY_EXPORT(ipred_smooth_v_16bpc);
HWY_EXPORT(ipred_smooth_h_16bpc);
HWY_EXPORT(ipred_paeth_16bpc);
HWY_EXPORT(ipred_filter_16bpc);
HWY_EXPORT(pal_pred_16bpc);
}  // namespace dav1d

namespace {
// Mirrors of Dav1dIntraPredDSPContext (src/ipred.h), so that this file does
// not need dav1d's bitdepth-templated C headers; the cfl entries and the h/v
// predictors are not installed here. Mode indices from src/levels.h:
// DC_PRED=0, LEFT_DC_PRED=3, TOP_DC_PRED=4, DC_128_PRED=5, Z1_PRED=6,
// Z2_PRED=7, Z3_PRED=8, SMOOTH_PRED=9, SMOOTH_V_PRED=10, SMOOTH_H_PRED=11,
// PAETH_PRED=12, FILTER_PRED=13; cfl_pred has DC_128_PRED+1=6 entries.
using AngularFn8 = void (*)(uint8_t *, ptrdiff_t, const uint8_t *,
                            int, int, int, int, int);
using CflAcFn8 = void (*)(int16_t *, const uint8_t *, ptrdiff_t,
                          int, int, int, int);
using CflPredFn8 = void (*)(uint8_t *, ptrdiff_t, const uint8_t *,
                            int, int, const int16_t *, int);
using PalPredFn8 = void (*)(uint8_t *, ptrdiff_t, const uint8_t *,
                            const uint8_t *, int, int);
struct IpredDSP8 {
    AngularFn8 intra_pred[14];
    CflAcFn8 cfl_ac[3];
    CflPredFn8 cfl_pred[6];
    PalPredFn8 pal_pred;
};

using AngularFn16 = void (*)(uint16_t *, ptrdiff_t, const uint16_t *,
                             int, int, int, int, int, int);
using CflAcFn16 = void (*)(int16_t *, const uint16_t *, ptrdiff_t,
                           int, int, int, int);
using CflPredFn16 = void (*)(uint16_t *, ptrdiff_t, const uint16_t *,
                             int, int, const int16_t *, int, int);
using PalPredFn16 = void (*)(uint16_t *, ptrdiff_t, const uint16_t *,
                             const uint8_t *, int, int);
struct IpredDSP16 {
    AngularFn16 intra_pred[14];
    CflAcFn16 cfl_ac[3];
    CflPredFn16 cfl_pred[6];
    PalPredFn16 pal_pred;
};
}  // namespace

namespace dav1d {

static void ipred_dsp_init_8bpc_hwy(void *const c) {
    auto *const ctx = static_cast<IpredDSP8 *>(c);
    ctx->intra_pred[0]  = HWY_DYNAMIC_POINTER(ipred_dc_8bpc);
    ctx->intra_pred[3]  = HWY_DYNAMIC_POINTER(ipred_dc_left_8bpc);
    ctx->intra_pred[4]  = HWY_DYNAMIC_POINTER(ipred_dc_top_8bpc);
    ctx->intra_pred[5]  = HWY_DYNAMIC_POINTER(ipred_dc_128_8bpc);
    ctx->intra_pred[6]  = HWY_DYNAMIC_POINTER(ipred_z1_8bpc);
    ctx->intra_pred[7]  = HWY_DYNAMIC_POINTER(ipred_z2_8bpc);
    ctx->intra_pred[8]  = HWY_DYNAMIC_POINTER(ipred_z3_8bpc);
    ctx->intra_pred[9]  = HWY_DYNAMIC_POINTER(ipred_smooth_8bpc);
    ctx->intra_pred[10] = HWY_DYNAMIC_POINTER(ipred_smooth_v_8bpc);
    ctx->intra_pred[11] = HWY_DYNAMIC_POINTER(ipred_smooth_h_8bpc);
    ctx->intra_pred[12] = HWY_DYNAMIC_POINTER(ipred_paeth_8bpc);
    ctx->intra_pred[13] = HWY_DYNAMIC_POINTER(ipred_filter_8bpc);
    ctx->pal_pred = HWY_DYNAMIC_POINTER(pal_pred_8bpc);
}

static void ipred_dsp_init_16bpc_hwy(void *const c) {
    auto *const ctx = static_cast<IpredDSP16 *>(c);
    ctx->intra_pred[0]  = HWY_DYNAMIC_POINTER(ipred_dc_16bpc);
    ctx->intra_pred[3]  = HWY_DYNAMIC_POINTER(ipred_dc_left_16bpc);
    ctx->intra_pred[4]  = HWY_DYNAMIC_POINTER(ipred_dc_top_16bpc);
    ctx->intra_pred[5]  = HWY_DYNAMIC_POINTER(ipred_dc_128_16bpc);
    ctx->intra_pred[6]  = HWY_DYNAMIC_POINTER(ipred_z1_16bpc);
    ctx->intra_pred[7]  = HWY_DYNAMIC_POINTER(ipred_z2_16bpc);
    ctx->intra_pred[8]  = HWY_DYNAMIC_POINTER(ipred_z3_16bpc);
    ctx->intra_pred[9]  = HWY_DYNAMIC_POINTER(ipred_smooth_16bpc);
    // 16bpc smooth_v is not covered (slot 10 keeps the C function).
    ctx->intra_pred[11] = HWY_DYNAMIC_POINTER(ipred_smooth_h_16bpc);
    ctx->intra_pred[12] = HWY_DYNAMIC_POINTER(ipred_paeth_16bpc);
    ctx->intra_pred[13] = HWY_DYNAMIC_POINTER(ipred_filter_16bpc);
    ctx->pal_pred = HWY_DYNAMIC_POINTER(pal_pred_16bpc);
}

}  // namespace dav1d

extern "C" void dav1d_ipred_dsp_init_hwy_8bpc(void *const c) {
    dav1d::hwy_init_chosen_target();
    dav1d::ipred_dsp_init_8bpc_hwy(c);
}

extern "C" void dav1d_ipred_dsp_init_hwy_16bpc(void *const c) {
    dav1d::hwy_init_chosen_target();
    dav1d::ipred_dsp_init_16bpc_hwy(c);
}

#endif  // HWY_ONCE
