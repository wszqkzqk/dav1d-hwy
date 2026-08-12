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
 * ON ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Film grain synthesis (src/filmgrain_tmpl.c) implemented with Google
// Highway: one source is compiled per SIMD target and the best one supported
// by the CPU is selected at runtime (HWY_DYNAMIC_DISPATCH). Bit-exact with
// the C code.

#include <stddef.h>
#include <stdint.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/hwy/filmgrain.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"

#include "dav1d/headers.h" // Dav1dFilmGrainData (public header, C++-clean)

// Defined in src/tables.c.
extern "C" const int16_t dav1d_gaussian_sequence[2048];

HWY_BEFORE_NAMESPACE();

namespace dav1d {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Matches ulog2() in include/common/intops.h.
static inline int hwy_ulog2(const unsigned v) {
#if defined(_MSC_VER) && !defined(__clang__)
    if (!v) return -1;
    unsigned long idx;
    _BitScanReverse(&idx, v);
    return (int) idx;
#else
    return v ? 31 - __builtin_clz(v) : -1;
#endif
}

static inline int hwy_imin(const int a, const int b) { return a < b ? a : b; }

// entry (grain LUT element) type per pixel type (src/filmgrain.h): int8_t
// for 8bpc, int16_t for 16bpc.
template <typename Pixel> struct FgEntryFor;
template <> struct FgEntryFor<uint8_t> { using type = int8_t; };
template <> struct FgEntryFor<uint16_t> { using type = int16_t; };

// GRAIN_WIDTH/GRAIN_HEIGHT/FG_BLOCK_SIZE (src/filmgrain.h) and
// SUB_GRAIN_WIDTH/SUB_GRAIN_HEIGHT (src/filmgrain_tmpl.c).
enum {
    kGrainW = 82,
    kGrainH = 73,
    kFGBlock = 32,
    kSubW = 44,
    kSubH = 38,
};

// Scalar helpers, exact copies from src/filmgrain_tmpl.c (get_random_number,
// round2, iclip). The LFSR state stays within 16 bits (the seed is 16-bit
// and the row mixing masks to 8 bits), so the int conversions never wrap.
static inline int fg_random_number(const int bits, unsigned *const state) {
    const int r = (int) *state;
    const unsigned bit = (unsigned) (((r >> 0) ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1);
    *state = (unsigned) ((r >> 1) | ((int) bit << 15));

    return (int) ((*state >> (16 - bits)) & ((1u << bits) - 1));
}

static inline int fg_round2(const int x, const int shift) {
    return (x + ((1 << shift) >> 1)) >> shift;
}

static inline int fg_iclip(const int v, const int lo, const int hi) {
    return v < lo ? lo : v > hi ? hi : v;
}

// Widens entry lanes (int8 for 8bpc, int16 for 16bpc) to int16/int32 lanes.
template <class D16, typename Entry>
static inline hn::VFromD<D16> hwy_load_entry16(const D16 d16,
                                               const Entry *const p,
                                               const int m) {
    if constexpr (sizeof(Entry) == 1) {
        const hn::Rebind<int8_t, D16> di8;
        return hn::PromoteTo(d16, hn::LoadN(di8, p, m));
    } else {
        return hn::LoadN(d16, p, m);
    }
}

template <class D32, typename Entry>
static inline hn::VFromD<D32> hwy_load_entry32(const D32 d32,
                                               const Entry *const p,
                                               const int m) {
    const hn::Rebind<Entry, D32> de;
    if constexpr (sizeof(Entry) == 1) {
        const hn::Rebind<int16_t, D32> di16;
        return hn::PromoteTo(d32, hn::PromoteTo(di16, hn::LoadN(de, p, m)));
    } else {
        return hn::PromoteTo(d32, hn::LoadN(de, p, m));
    }
}

// Widens pixel lanes (uint8/uint16, values <= 4095) to int32 lanes.
template <class D32, typename Pixel>
static inline hn::VFromD<D32> hwy_load_px32(const D32 d32, const Pixel *const p,
                                            const int m) {
    const hn::Rebind<Pixel, D32> dp;
    if constexpr (sizeof(Pixel) == 1) {
        const hn::Rebind<uint16_t, D32> du16;
        return hn::PromoteTo(d32, hn::PromoteTo(du16, hn::LoadN(dp, p, m)));
    } else {
        return hn::PromoteTo(d32, hn::LoadN(dp, p, m));
    }
}

// psum[i] = the AR contributions of the rows above y (all taps with dy < 0),
// in the C coefficient order (dy outer, dx inner). i32 accumulation cannot
// wrap: |psum| <= 21 * 128 * 32767 < 2^31 even for arbitrary stored entries.
// WidenMulAccumulate's low/high are the in-order lane halves, so the split
// stores below preserve positions.
template <typename Entry>
static void hwy_fg_ar_psum(int32_t *const psum, const Entry buf[][kGrainW],
                           const int8_t *coeff, const int lag, const int y,
                           const int n)
{
    const hn::ScalableTag<int32_t> d32;
    const hn::Repartition<int16_t, decltype(d32)> d16;
    const int L32 = (int) hn::Lanes(d32);
    const int L = (int) hn::Lanes(d16);
    for (int x = 0; x < n; x += L) {
        const int m = x + L <= n ? L : n - x;
        auto lo = hn::Zero(d32), hi = hn::Zero(d32);
        const int8_t *c = coeff;
        for (int dy = -lag; dy < 0; dy++)
            for (int dx = -lag; dx <= lag; dx++) {
                const auto v = hwy_load_entry16(d16, buf[y + dy] + 3 + x + dx,
                                                m);
                lo = hn::WidenMulAccumulate(d32, v, hn::Set(d16, (int16_t) *c++),
                                            lo, hi);
            }
        const int m0 = hwy_imin(m, L32);
        hn::StoreN(lo, d32, psum + x, m0);
        if (m > m0) hn::StoreN(hi, d32, psum + x + L32, m - m0);
    }
}

// Autoregressive filtering shared by generate_grain_y/uv_c(). The
// previous-row taps (the bulk: lag*(2*lag+1) of the 2*lag*(lag+1)
// coefficients) are vectorized into psum; the same-row recurrence and the
// chroma luma contribution are inherently serial and stay scalar, as in C.
// lag == 0 without a luma tap reduces to a clip (round2(0, s) == 0); note
// the chroma (0,0) luma tap still exists at lag == 0 when num_y_points > 0.
template <typename Entry, int SX, int SY, bool kUV>
static void hwy_fg_ar(Entry buf[][kGrainW], const Entry (*buf_y)[kGrainW],
                      const Dav1dFilmGrainData *const data, const intptr_t uv,
                      const int w, const int h, const int gmin, const int gmax)
{
    const int lag = data->ar_coeff_lag;
    const int8_t *const coeffs = kUV ? data->ar_coeffs_uv[uv] : data->ar_coeffs_y;
    const int ar_shift = (int) data->ar_coeff_shift;
    const int n = w - 2 * 3; // ar_pad = 3

    if (!lag && !(kUV && data->num_y_points)) {
        const hn::ScalableTag<Entry> de;
        const int L = (int) hn::Lanes(de);
        const auto vlo = hn::Set(de, (Entry) gmin);
        const auto vhi = hn::Set(de, (Entry) gmax);
        for (int y = 3; y < h; y++)
            for (int x = 3; x < w - 3; x += L) {
                const int m = x + L <= w - 3 ? L : w - 3 - x;
                hn::StoreN(hn::Clamp(hn::LoadN(de, buf[y] + x, m), vlo, vhi),
                           de, buf[y] + x, m);
            }
        return;
    }

    int32_t psum[kGrainW]; // n <= 76
    for (int y = 3; y < h; y++) {
        hwy_fg_ar_psum(psum, buf, coeffs, lag, y, n);
        const int8_t *const rc = coeffs + lag * (2 * lag + 1); // dy == 0 taps
        for (int x = 3; x < w - 3; x++) {
            int sum = psum[x - 3];
            for (int dx = -lag; dx < 0; dx++)
                sum += rc[lag + dx] * buf[y][x + dx];
            if (kUV && data->num_y_points) {
                int luma = 0;
                const int lx = ((x - 3) << SX) + 3;
                const int ly = ((y - 3) << SY) + 3;
                for (int i = 0; i <= SY; i++)
                    for (int j = 0; j <= SX; j++)
                        luma += buf_y[ly + i][lx + j];
                sum += fg_round2(luma, SX + SY) * coeffs[2 * lag * (lag + 1)];
            }
            const int grain = buf[y][x] + fg_round2(sum, ar_shift);
            buf[y][x] = (Entry) fg_iclip(grain, gmin, gmax);
        }
    }
}

// Port of generate_grain_y_c(). The gaussian fill is a serial LFSR plus a
// table lookup and stays scalar; the (Entry) cast reproduces the implicit
// conversion of the C code (which wraps for extreme 8bpc fills).
template <typename Entry>
static void hwy_generate_grain_y(Entry buf[][kGrainW],
                                 const Dav1dFilmGrainData *const data,
                                 const int bitdepth_min_8)
{
    const int shift = 4 - bitdepth_min_8 + data->grain_scale_shift;
    const int grain_ctr = 128 << bitdepth_min_8;
    unsigned seed = data->seed;

    for (int y = 0; y < kGrainH; y++)
        for (int x = 0; x < kGrainW; x++)
            buf[y][x] = (Entry) fg_round2(
                dav1d_gaussian_sequence[fg_random_number(11, &seed)], shift);

    hwy_fg_ar<Entry, 0, 0, false>(buf, (const Entry (*)[kGrainW]) nullptr,
                                  data, 0, kGrainW, kGrainH,
                                  -grain_ctr, grain_ctr - 1);
}

// Port of generate_grain_uv_c().
template <typename Entry, int SX, int SY>
static void hwy_generate_grain_uv(Entry buf[][kGrainW],
                                  const Entry buf_y[][kGrainW],
                                  const Dav1dFilmGrainData *const data,
                                  const intptr_t uv, const int bitdepth_min_8)
{
    const int shift = 4 - bitdepth_min_8 + data->grain_scale_shift;
    const int grain_ctr = 128 << bitdepth_min_8;
    unsigned seed = data->seed ^ (uv ? 0x49d8 : 0xb524);
    const int w = SX ? kSubW : kGrainW, h = SY ? kSubH : kGrainH;

    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            buf[y][x] = (Entry) fg_round2(
                dav1d_gaussian_sequence[fg_random_number(11, &seed)], shift);

    hwy_fg_ar<Entry, SX, SY, true>(buf, buf_y, data, uv, w, h,
                                   -grain_ctr, grain_ctr - 1);
}

// Exact copy of sample_lut() from src/filmgrain_tmpl.c. All reachable
// indices stay within the [GRAIN_HEIGHT + 1][GRAIN_WIDTH] LUT: by == 1 only
// occurs for y < 2 (resp. 1 subsampled), bx == 1 only for x < 2 (resp. 1).
template <typename Entry>
static inline Entry fg_sample_lut(const Entry grain_lut[][kGrainW],
                                  const int offsets[2][2], const int subx,
                                  const int suby, const int bx, const int by,
                                  const int x, const int y)
{
    const int randval = offsets[bx][by];
    const int offx = 3 + (2 >> subx) * (3 + (randval >> 4));
    const int offy = 3 + (2 >> suby) * (3 + (randval & 0xF));
    return grain_lut[offy + y + (kFGBlock >> suby) * by]
                    [offx + x + (kFGBlock >> subx) * bx];
}

// dst[i] = iclip(src[i] + round2(sbuf[i] * grain[i], shift), minv, maxv) for
// i in [0, n); the shared tail of add_noise_y/add_noise_uv.
//
// 8bpc: s*g fits int16 (|255 * -128| = 32640) but adding the rounding
// constant could overflow it, so round2(p, s) is evaluated exactly as
// (p >> s) + (((p & (2^s - 1)) + r) >> s): with the arithmetic (floor)
// shift p = (p >> s) * 2^s + (p & (2^s - 1)) with the mask term in
// [0, 2^s), so the second addend is in {0, 1} and no intermediate overflows
// int16. 16bpc: |s*g| <= 255 * 2048 needs int32 lanes.
template <typename Pixel, typename Entry>
static void hwy_fg_noise_row(Pixel *const dst, const Pixel *const src,
                             const Entry *const grain,
                             const uint8_t *const sbuf, const int n,
                             const int shift, const int minv, const int maxv)
{
    if constexpr (sizeof(Pixel) == 1) {
        const hn::ScalableTag<int16_t> d16;
        const hn::Rebind<uint8_t, decltype(d16)> du8;
        const hn::Rebind<int8_t, decltype(d16)> di8;
        const int L = (int) hn::Lanes(d16);
        const auto vrnd = hn::Set(d16, 1 << (shift - 1));
        const auto vmask = hn::Set(d16, (1 << shift) - 1);
        const auto vmin = hn::Set(d16, minv), vmax = hn::Set(d16, maxv);
        for (int x = 0; x < n; x += L) {
            const int m = x + L <= n ? L : n - x;
            const auto s = hn::PromoteTo(d16, hn::LoadN(du8, sbuf + x, m));
            const auto g = hn::PromoteTo(d16, hn::LoadN(di8, grain + x, m));
            const auto p = hn::Mul(s, g);
            const auto noise = hn::Add(hn::ShiftRightSame(p, shift),
                hn::ShiftRightSame(hn::Add(hn::And(p, vmask), vrnd), shift));
            const auto px = hn::PromoteTo(d16, hn::LoadN(du8, src + x, m));
            // Clamped to [minv, maxv] (within [0, 255]), so the narrowing
            // demotion is exact.
            hn::StoreN(hn::DemoteTo(du8, hn::Clamp(hn::Add(px, noise), vmin,
                                                   vmax)),
                       du8, dst + x, m);
        }
    } else {
        const hn::ScalableTag<int32_t> d32;
        const hn::Rebind<uint16_t, decltype(d32)> du16;
        const hn::Rebind<uint8_t, decltype(d32)> du8;
        const hn::Rebind<int16_t, decltype(d32)> di16;
        const int L = (int) hn::Lanes(d32);
        const auto vrnd = hn::Set(d32, 1 << (shift - 1));
        const auto vmin = hn::Set(d32, minv), vmax = hn::Set(d32, maxv);
        for (int x = 0; x < n; x += L) {
            const int m = x + L <= n ? L : n - x;
            const auto s = hn::PromoteTo(d32,
                hn::PromoteTo(du16, hn::LoadN(du8, sbuf + x, m)));
            const auto g = hn::PromoteTo(d32, hn::LoadN(di16, grain + x, m));
            const auto noise = hn::ShiftRightSame(hn::Add(hn::Mul(s, g), vrnd),
                                                  shift);
            const auto px = hn::PromoteTo(d32, hn::LoadN(du16, src + x, m));
            hn::StoreN(hn::DemoteTo(du16, hn::Clamp(hn::Add(px, noise), vmin,
                                                    vmax)),
                       du16, dst + x, m);
        }
    }
}

// The scaling LUT is indexed by the pixel value (up to 4096 entries), so the
// lookup stays a scalar gather on every target; the arithmetic around it is
// vectorized. n <= 32 (one FG block row, subsampled at most 16).
template <typename Pixel, typename Entry>
static void hwy_fgy_row(Pixel *const dst, const Pixel *const src,
                        const Entry *const grain, const uint8_t *const scaling,
                        const int n, const int shift, const int minv,
                        const int maxv)
{
    uint8_t sbuf[kFGBlock];
    for (int i = 0; i < n; i++)
        sbuf[i] = scaling[src[i]];
    hwy_fg_noise_row(dst, src, grain, sbuf, n, shift, minv, maxv);
}

// out[i] = iclip(round2(old[i]*wo + new[i]*wn, 5), gmin, gmax): the blended
// grain of an overlapped block edge, weights constant per row. int32 lanes:
// |old*27 + new*17| <= 44 * 2048 overflows int16 for 16bpc. After clamping
// to [gmin, gmax] (within the entry range) the demotion is exact.
template <typename Entry>
static void hwy_fg_blend_row(Entry *const dst, const Entry *const old_row,
                             const Entry *const new_row, const int wo,
                             const int wn, const int n, const int gmin,
                             const int gmax)
{
    const hn::ScalableTag<int32_t> d32;
    const hn::Rebind<Entry, decltype(d32)> de;
    const int L = (int) hn::Lanes(d32);
    const auto vwo = hn::Set(d32, wo), vwn = hn::Set(d32, wn);
    const auto vrnd = hn::Set(d32, 1 << 4);
    const auto vmin = hn::Set(d32, gmin), vmax = hn::Set(d32, gmax);
    for (int x = 0; x < n; x += L) {
        const int m = x + L <= n ? L : n - x;
        const auto vo = hwy_load_entry32(d32, old_row + x, m);
        const auto vn = hwy_load_entry32(d32, new_row + x, m);
        const auto g = hn::Clamp(hn::ShiftRight<5>(hn::Add(hn::Add(
            hn::Mul(vo, vwo), hn::Mul(vn, vwn)), vrnd)), vmin, vmax);
        hn::StoreN(hn::DemoteTo(de, g), de, dst + x, m);
    }
}

// Exact copy of the add_noise_y() macro from src/filmgrain_tmpl.c.
template <typename Pixel>
static inline void hwy_add_noise_y(Pixel *const dst_row,
                                   const Pixel *const src_row,
                                   const ptrdiff_t pxstride,
                                   const Dav1dFilmGrainData *const data,
                                   const uint8_t *const scaling,
                                   const int minv, const int maxv,
                                   const int bx, const int x, const int y,
                                   const int grain)
{
    const Pixel *const src = src_row + y * pxstride + x + bx;
    Pixel *const dst = dst_row + y * pxstride + x + bx;
    const int noise = fg_round2(scaling[*src] * grain, data->scaling_shift);
    *dst = (Pixel) fg_iclip(*src + noise, minv, maxv);
}

// Exact copy of the add_noise_uv() macro from src/filmgrain_tmpl.c.
template <typename Pixel, int SX, int SY>
static inline void hwy_add_noise_uv(Pixel *const dst_row,
                                    const Pixel *const src_row,
                                    const ptrdiff_t pxstride,
                                    const Pixel *const luma_row,
                                    const ptrdiff_t luma_pxstride,
                                    const Dav1dFilmGrainData *const data,
                                    const uint8_t *const scaling, const int uv,
                                    const int minv, const int maxv,
                                    const int bitdepth_min_8, const int bdmax,
                                    const int bx, const int x, const int y,
                                    const int grain)
{
    const int lx = (bx + x) << SX;
    const int ly = y << SY;
    const Pixel *const luma = luma_row + ly * luma_pxstride + lx;
    Pixel avg = luma[0];
    if (SX)
        avg = (Pixel) ((avg + luma[1] + 1) >> 1);
    const Pixel *const src = src_row + y * pxstride + bx + x;
    Pixel *const dst = dst_row + y * pxstride + bx + x;
    int val = avg;
    if (!data->chroma_scaling_from_luma) {
        const int combined = avg * data->uv_luma_mult[uv] +
                             *src * data->uv_mult[uv];
        val = fg_iclip((combined >> 6) +
                       data->uv_offset[uv] * (1 << bitdepth_min_8), 0, bdmax);
    }
    const int noise = fg_round2(scaling[val] * grain, data->scaling_shift);
    *dst = (Pixel) fg_iclip(*src + noise, minv, maxv);
}

// Port of fgy_32x32xn_c(). Per block the LUT samples of a row are
// contiguous, so the straight regions run vectorized; the overlapped edge
// rows use a constant-weight blend (vectorized) and the 1-2 wide overlapped
// columns/corner stay scalar, exactly as C.
template <typename Pixel, typename Entry>
static void hwy_fgy(Pixel *const dst_row, const Pixel *const src_row,
                    const ptrdiff_t stride,
                    const Dav1dFilmGrainData *const data, const size_t pw,
                    const uint8_t *const scaling,
                    const Entry grain_lut[][kGrainW],
                    const int bh, const int row_num, const int bitdepth_min_8)
{
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);
    const int rows = 1 + (data->overlap_flag && row_num > 0);
    const int grain_ctr = 128 << bitdepth_min_8;
    const int grain_min = -grain_ctr, grain_max = grain_ctr - 1;

    int min_value, max_value;
    if (data->clip_to_restricted_range) {
        min_value = 16 << bitdepth_min_8;
        max_value = 235 << bitdepth_min_8;
    } else {
        min_value = 0;
        max_value = (1 << (8 + bitdepth_min_8)) - 1;
    }

    // seed[0] contains the current row, seed[1] contains the previous
    unsigned seed[2];
    for (int i = 0; i < rows; i++) {
        seed[i] = data->seed;
        seed[i] ^= (unsigned) ((((row_num - i) * 37  + 178) & 0xFF) << 8);
        seed[i] ^= (unsigned) (((row_num - i) * 173 + 105) & 0xFF);
    }

    int offsets[2 /* col offset */][2 /* row offset */];
    Entry gtmp[kFGBlock];

    // process this row in FG_BLOCK_SIZE^2 blocks
    for (unsigned bx = 0; bx < pw; bx += kFGBlock) {
        const int bw = hwy_imin(kFGBlock, (int) pw - (int) bx);

        if (data->overlap_flag && bx) {
            // shift previous offsets left
            for (int i = 0; i < rows; i++)
                offsets[1][i] = offsets[0][i];
        }

        // update current offsets
        for (int i = 0; i < rows; i++)
            offsets[0][i] = fg_random_number(8, &seed[i]);

        // x/y block offsets to compensate for overlapped regions
        const int ystart = data->overlap_flag && row_num ? hwy_imin(2, bh) : 0;
        const int xstart = data->overlap_flag && bx      ? hwy_imin(2, bw) : 0;

        static const int w[2][2] = { { 27, 17 }, { 17, 27 } };

        const int offx0 = 3 + 2 * (3 + (offsets[0][0] >> 4));
        const int offy0 = 3 + 2 * (3 + (offsets[0][0] & 0xF));

        for (int y = ystart; y < bh; y++) {
            // Non-overlapped image region (straightforward)
            const int n = bw - xstart;
            if (n > 0)
                hwy_fgy_row(dst_row + y * pxstride + bx + xstart,
                            src_row + y * pxstride + bx + xstart,
                            &grain_lut[offy0 + y][offx0 + xstart], scaling, n,
                            data->scaling_shift, min_value, max_value);

            // Special case for overlapped column
            for (int x = 0; x < xstart; x++) {
                int grain = fg_sample_lut(grain_lut, offsets, 0, 0, 0, 0, x, y);
                const int old = fg_sample_lut(grain_lut, offsets, 0, 0, 1, 0, x, y);
                grain = fg_round2(old * w[x][0] + grain * w[x][1], 5);
                grain = fg_iclip(grain, grain_min, grain_max);
                hwy_add_noise_y(dst_row, src_row, pxstride, data, scaling,
                                min_value, max_value, (int) bx, x, y, grain);
            }
        }

        for (int y = 0; y < ystart; y++) {
            // Special case for overlapped row (sans corner)
            const int offx1 = 3 + 2 * (3 + (offsets[0][1] >> 4));
            const int offy1 = 3 + 2 * (3 + (offsets[0][1] & 0xF));
            hwy_fg_blend_row(gtmp, &grain_lut[offy1 + y + kFGBlock][offx1],
                             &grain_lut[offy0 + y][offx0], w[y][0], w[y][1],
                             bw, grain_min, grain_max);
            const int n = bw - xstart;
            if (n > 0)
                hwy_fgy_row(dst_row + y * pxstride + bx + xstart,
                            src_row + y * pxstride + bx + xstart,
                            gtmp + xstart, scaling, n, data->scaling_shift,
                            min_value, max_value);

            // Special case for doubly-overlapped corner
            for (int x = 0; x < xstart; x++) {
                // Blend the top pixel with the top left block
                int top = fg_sample_lut(grain_lut, offsets, 0, 0, 0, 1, x, y);
                int old = fg_sample_lut(grain_lut, offsets, 0, 0, 1, 1, x, y);
                top = fg_round2(old * w[x][0] + top * w[x][1], 5);
                top = fg_iclip(top, grain_min, grain_max);

                // Blend the current pixel with the left block
                int grain = fg_sample_lut(grain_lut, offsets, 0, 0, 0, 0, x, y);
                old = fg_sample_lut(grain_lut, offsets, 0, 0, 1, 0, x, y);
                grain = fg_round2(old * w[x][0] + grain * w[x][1], 5);
                grain = fg_iclip(grain, grain_min, grain_max);

                // Mix the two rows together and apply grain
                grain = fg_round2(top * w[y][0] + grain * w[y][1], 5);
                grain = fg_iclip(grain, grain_min, grain_max);
                hwy_add_noise_y(dst_row, src_row, pxstride, data, scaling,
                                min_value, max_value, (int) bx, x, y, grain);
            }
        }
    }
}

// val[i] = iclip((avg[i]*lm + src[i]*um) >> 6 + off, 0, bdmax): the chroma
// scaling index when !chroma_scaling_from_luma. int32 lanes: the products
// reach 128 * 4095 (16bpc), exceeding int16. (combined >> 6) is an
// arithmetic shift, as in C.
template <typename Pixel>
static void hwy_fguv_val(Pixel *const vals, const Pixel *const avg,
                         const Pixel *const src, const int n, const int lm,
                         const int um, const int off, const int bdmax)
{
    const hn::ScalableTag<int32_t> d32;
    const hn::Rebind<Pixel, decltype(d32)> dp;
    const int L = (int) hn::Lanes(d32);
    const auto vlm = hn::Set(d32, lm), vum = hn::Set(d32, um);
    const auto voff = hn::Set(d32, off);
    const auto vzero = hn::Zero(d32), vmax = hn::Set(d32, bdmax);
    for (int x = 0; x < n; x += L) {
        const int m = x + L <= n ? L : n - x;
        const auto a = hwy_load_px32(d32, avg + x, m);
        const auto s = hwy_load_px32(d32, src + x, m);
        const auto val = hn::Clamp(hn::Add(hn::ShiftRight<6>(hn::Add(
            hn::Mul(a, vlm), hn::Mul(s, vum))), voff), vzero, vmax);
        // Clamped to [0, bdmax], so the narrowing demotion is exact.
        hn::StoreN(hn::DemoteTo(dp, val), dp, vals + x, m);
    }
}

// The luma average of add_noise_uv(): the (l0 + l1 + 1) >> 1 pair average
// for subsampled chroma (exactly AverageRound), plain luma otherwise.
template <typename Pixel, int SX>
static void hwy_luma_avg(Pixel *const out, const Pixel *const luma,
                         const int n)
{
    const hn::ScalableTag<Pixel> dp;
    const int L = (int) hn::Lanes(dp);
    int x = 0;
    if constexpr (SX != 0) {
        for (; x + L <= n; x += L) {
            hn::VFromD<decltype(dp)> a, b;
            hn::LoadInterleaved2(dp, luma + 2 * x, a, b);
            hn::StoreU(hn::AverageRound(a, b), dp, out + x);
        }
        for (; x < n; x++)
            out[x] = (Pixel) ((luma[2 * x] + luma[2 * x + 1] + 1) >> 1);
    } else {
        for (; x + L <= n; x += L)
            hn::StoreU(hn::LoadU(dp, luma + x), dp, out + x);
        if (x < n)
            hn::StoreN(hn::LoadN(dp, luma + x, n - x), dp, out + x, n - x);
    }
}

template <typename Pixel, typename Entry, int SX>
static void hwy_fguv_row(Pixel *const dst, const Pixel *const src,
                         const Pixel *const luma, const Entry *const grain,
                         const uint8_t *const scaling, const int n,
                         const Dav1dFilmGrainData *const data, const int uv,
                         const int minv, const int maxv,
                         const int bitdepth_min_8, const int bdmax)
{
    Pixel vals[kFGBlock];
    if (data->chroma_scaling_from_luma) {
        hwy_luma_avg<Pixel, SX>(vals, luma, n);
    } else {
        Pixel avg[kFGBlock];
        hwy_luma_avg<Pixel, SX>(avg, luma, n);
        hwy_fguv_val(vals, avg, src, n, data->uv_luma_mult[uv],
                     data->uv_mult[uv],
                     data->uv_offset[uv] * (1 << bitdepth_min_8), bdmax);
    }
    uint8_t sbuf[kFGBlock];
    for (int i = 0; i < n; i++)
        sbuf[i] = scaling[vals[i]];
    hwy_fg_noise_row(dst, src, grain, sbuf, n, data->scaling_shift, minv,
                     maxv);
}

// Port of fguv_32x32xn_c(); same vectorization structure as hwy_fgy().
template <typename Pixel, int SX, int SY>
static void hwy_fguv(Pixel *const dst_row, const Pixel *const src_row,
                     const ptrdiff_t stride,
                     const Dav1dFilmGrainData *const data, const size_t pw,
                     const uint8_t *const scaling,
                     const typename FgEntryFor<Pixel>::type grain_lut[][kGrainW],
                     const int bh, const int row_num, const Pixel *const luma_row,
                     const ptrdiff_t luma_stride, const int uv, const int is_id,
                     const int bitdepth_min_8)
{
    using Entry = typename FgEntryFor<Pixel>::type;
    const ptrdiff_t pxstride = stride / (ptrdiff_t) sizeof(Pixel);
    const ptrdiff_t luma_pxstride = luma_stride / (ptrdiff_t) sizeof(Pixel);
    const int rows = 1 + (data->overlap_flag && row_num > 0);
    const int grain_ctr = 128 << bitdepth_min_8;
    const int grain_min = -grain_ctr, grain_max = grain_ctr - 1;
    const int bdmax = (1 << (8 + bitdepth_min_8)) - 1;

    int min_value, max_value;
    if (data->clip_to_restricted_range) {
        min_value = 16 << bitdepth_min_8;
        max_value = (is_id ? 235 : 240) << bitdepth_min_8;
    } else {
        min_value = 0;
        max_value = bdmax;
    }

    // seed[0] contains the current row, seed[1] contains the previous
    unsigned seed[2];
    for (int i = 0; i < rows; i++) {
        seed[i] = data->seed;
        seed[i] ^= (unsigned) ((((row_num - i) * 37  + 178) & 0xFF) << 8);
        seed[i] ^= (unsigned) (((row_num - i) * 173 + 105) & 0xFF);
    }

    int offsets[2 /* col offset */][2 /* row offset */];
    Entry gtmp[kFGBlock];

    // process this row in FG_BLOCK_SIZE^2 blocks (subsampled)
    for (unsigned bx = 0; bx < pw; bx += kFGBlock >> SX) {
        const int bw = hwy_imin(kFGBlock >> SX, (int) (pw - bx));

        if (data->overlap_flag && bx) {
            // shift previous offsets left
            for (int i = 0; i < rows; i++)
                offsets[1][i] = offsets[0][i];
        }

        // update current offsets
        for (int i = 0; i < rows; i++)
            offsets[0][i] = fg_random_number(8, &seed[i]);

        // x/y block offsets to compensate for overlapped regions
        const int ystart = data->overlap_flag && row_num ? hwy_imin(2 >> SY, bh) : 0;
        const int xstart = data->overlap_flag && bx      ? hwy_imin(2 >> SX, bw) : 0;

        static const int w[2 /* sub */][2 /* off */][2] = {
            { { 27, 17 }, { 17, 27 } },
            { { 23, 22 } },
        };

        const int offx0 = 3 + (2 >> SX) * (3 + (offsets[0][0] >> 4));
        const int offy0 = 3 + (2 >> SY) * (3 + (offsets[0][0] & 0xF));

        for (int y = ystart; y < bh; y++) {
            // Non-overlapped image region (straightforward)
            const int n = bw - xstart;
            if (n > 0)
                hwy_fguv_row<Pixel, Entry, SX>(
                    dst_row + y * pxstride + bx + xstart,
                    src_row + y * pxstride + bx + xstart,
                    luma_row + (y << SY) * luma_pxstride + ((bx + xstart) << SX),
                    &grain_lut[offy0 + y][offx0 + xstart], scaling, n, data,
                    uv, min_value, max_value, bitdepth_min_8, bdmax);

            // Special case for overlapped column
            for (int x = 0; x < xstart; x++) {
                int grain = fg_sample_lut(grain_lut, offsets, SX, SY, 0, 0, x, y);
                const int old = fg_sample_lut(grain_lut, offsets, SX, SY, 1, 0, x, y);
                grain = fg_round2(old * w[SX][x][0] + grain * w[SX][x][1], 5);
                grain = fg_iclip(grain, grain_min, grain_max);
                hwy_add_noise_uv<Pixel, SX, SY>(dst_row, src_row, pxstride,
                    luma_row, luma_pxstride, data, scaling, uv, min_value,
                    max_value, bitdepth_min_8, bdmax, (int) bx, x, y, grain);
            }
        }

        for (int y = 0; y < ystart; y++) {
            // Special case for overlapped row (sans corner)
            const int offx1 = 3 + (2 >> SX) * (3 + (offsets[0][1] >> 4));
            const int offy1 = 3 + (2 >> SY) * (3 + (offsets[0][1] & 0xF));
            hwy_fg_blend_row(gtmp, &grain_lut[offy1 + y + (kFGBlock >> SY)][offx1],
                             &grain_lut[offy0 + y][offx0], w[SY][y][0],
                             w[SY][y][1], bw, grain_min, grain_max);
            const int n = bw - xstart;
            if (n > 0)
                hwy_fguv_row<Pixel, Entry, SX>(
                    dst_row + y * pxstride + bx + xstart,
                    src_row + y * pxstride + bx + xstart,
                    luma_row + (y << SY) * luma_pxstride + ((bx + xstart) << SX),
                    gtmp + xstart, scaling, n, data, uv, min_value, max_value,
                    bitdepth_min_8, bdmax);

            // Special case for doubly-overlapped corner
            for (int x = 0; x < xstart; x++) {
                // Blend the top pixel with the top left block
                int top = fg_sample_lut(grain_lut, offsets, SX, SY, 0, 1, x, y);
                int old = fg_sample_lut(grain_lut, offsets, SX, SY, 1, 1, x, y);
                top = fg_round2(old * w[SX][x][0] + top * w[SX][x][1], 5);
                top = fg_iclip(top, grain_min, grain_max);

                // Blend the current pixel with the left block
                int grain = fg_sample_lut(grain_lut, offsets, SX, SY, 0, 0, x, y);
                old = fg_sample_lut(grain_lut, offsets, SX, SY, 1, 0, x, y);
                grain = fg_round2(old * w[SX][x][0] + grain * w[SX][x][1], 5);
                grain = fg_iclip(grain, grain_min, grain_max);

                // Mix the two rows together and apply to image
                grain = fg_round2(top * w[SY][y][0] + grain * w[SY][y][1], 5);
                grain = fg_iclip(grain, grain_min, grain_max);
                hwy_add_noise_uv<Pixel, SX, SY>(dst_row, src_row, pxstride,
                    luma_row, luma_pxstride, data, scaling, uv, min_value,
                    max_value, bitdepth_min_8, bdmax, (int) bx, x, y, grain);
            }
        }
    }
}

#define FG_FNS(bpc, sfx) \
void generate_grain_y_##sfx(int##bpc##_t buf[][kGrainW], \
                            const Dav1dFilmGrainData *const data \
                            HIGHBD_SUFFIX(bpc)) \
{ \
    hwy_generate_grain_y(buf, data, BD_MINUS_8(bpc)); \
} \
void generate_grain_uv_420_##sfx(int##bpc##_t buf[][kGrainW], \
                                 const int##bpc##_t buf_y[][kGrainW], \
                                 const Dav1dFilmGrainData *const data, \
                                 const intptr_t uv HIGHBD_SUFFIX(bpc)) \
{ \
    hwy_generate_grain_uv<int##bpc##_t, 1, 1>(buf, buf_y, data, uv, \
                                              BD_MINUS_8(bpc)); \
} \
void generate_grain_uv_422_##sfx(int##bpc##_t buf[][kGrainW], \
                                 const int##bpc##_t buf_y[][kGrainW], \
                                 const Dav1dFilmGrainData *const data, \
                                 const intptr_t uv HIGHBD_SUFFIX(bpc)) \
{ \
    hwy_generate_grain_uv<int##bpc##_t, 1, 0>(buf, buf_y, data, uv, \
                                              BD_MINUS_8(bpc)); \
} \
void generate_grain_uv_444_##sfx(int##bpc##_t buf[][kGrainW], \
                                 const int##bpc##_t buf_y[][kGrainW], \
                                 const Dav1dFilmGrainData *const data, \
                                 const intptr_t uv HIGHBD_SUFFIX(bpc)) \
{ \
    hwy_generate_grain_uv<int##bpc##_t, 0, 0>(buf, buf_y, data, uv, \
                                              BD_MINUS_8(bpc)); \
} \
void fgy_32x32xn_##sfx(uint##bpc##_t *const dst_row, \
                       const uint##bpc##_t *const src_row, \
                       const ptrdiff_t stride, \
                       const Dav1dFilmGrainData *const data, const size_t pw, \
                       const uint8_t *const scaling, \
                       const int##bpc##_t grain_lut[][kGrainW], \
                       const int bh, const int row_num HIGHBD_SUFFIX(bpc)) \
{ \
    hwy_fgy(dst_row, src_row, stride, data, pw, scaling, grain_lut, bh, \
            row_num, BD_MINUS_8(bpc)); \
} \
void fguv_32x32xn_420_##sfx(uint##bpc##_t *const dst_row, \
                            const uint##bpc##_t *const src_row, \
                            const ptrdiff_t stride, \
                            const Dav1dFilmGrainData *const data, \
                            const size_t pw, const uint8_t *const scaling, \
                            const int##bpc##_t grain_lut[][kGrainW], \
                            const int bh, const int row_num, \
                            const uint##bpc##_t *const luma_row, \
                            const ptrdiff_t luma_stride, const int uv_pl, \
                            const int is_id HIGHBD_SUFFIX(bpc)) \
{ \
    hwy_fguv<uint##bpc##_t, 1, 1>(dst_row, src_row, stride, data, pw, scaling, \
                                  grain_lut, bh, row_num, luma_row, \
                                  luma_stride, uv_pl, is_id, BD_MINUS_8(bpc)); \
} \
void fguv_32x32xn_422_##sfx(uint##bpc##_t *const dst_row, \
                            const uint##bpc##_t *const src_row, \
                            const ptrdiff_t stride, \
                            const Dav1dFilmGrainData *const data, \
                            const size_t pw, const uint8_t *const scaling, \
                            const int##bpc##_t grain_lut[][kGrainW], \
                            const int bh, const int row_num, \
                            const uint##bpc##_t *const luma_row, \
                            const ptrdiff_t luma_stride, const int uv_pl, \
                            const int is_id HIGHBD_SUFFIX(bpc)) \
{ \
    hwy_fguv<uint##bpc##_t, 1, 0>(dst_row, src_row, stride, data, pw, scaling, \
                                  grain_lut, bh, row_num, luma_row, \
                                  luma_stride, uv_pl, is_id, BD_MINUS_8(bpc)); \
} \
void fguv_32x32xn_444_##sfx(uint##bpc##_t *const dst_row, \
                            const uint##bpc##_t *const src_row, \
                            const ptrdiff_t stride, \
                            const Dav1dFilmGrainData *const data, \
                            const size_t pw, const uint8_t *const scaling, \
                            const int##bpc##_t grain_lut[][kGrainW], \
                            const int bh, const int row_num, \
                            const uint##bpc##_t *const luma_row, \
                            const ptrdiff_t luma_stride, const int uv_pl, \
                            const int is_id HIGHBD_SUFFIX(bpc)) \
{ \
    hwy_fguv<uint##bpc##_t, 0, 0>(dst_row, src_row, stride, data, pw, scaling, \
                                  grain_lut, bh, row_num, luma_row, \
                                  luma_stride, uv_pl, is_id, BD_MINUS_8(bpc)); \
}

#define HIGHBD_SUFFIX(bpc)
#define BD_MINUS_8(bpc) 0
FG_FNS(8, 8bpc)
#undef HIGHBD_SUFFIX
#undef BD_MINUS_8
#define HIGHBD_SUFFIX(bpc) , const int bitdepth_max
#define BD_MINUS_8(bpc) hwy_ulog2((unsigned) bitdepth_max) - 7
FG_FNS(16, 16bpc)
#undef HIGHBD_SUFFIX
#undef BD_MINUS_8
#undef FG_FNS

}  // namespace HWY_NAMESPACE
}  // namespace dav1d

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace dav1d {
HWY_EXPORT(generate_grain_y_8bpc);
HWY_EXPORT(generate_grain_uv_420_8bpc);
HWY_EXPORT(generate_grain_uv_422_8bpc);
HWY_EXPORT(generate_grain_uv_444_8bpc);
HWY_EXPORT(fgy_32x32xn_8bpc);
HWY_EXPORT(fguv_32x32xn_420_8bpc);
HWY_EXPORT(fguv_32x32xn_422_8bpc);
HWY_EXPORT(fguv_32x32xn_444_8bpc);
HWY_EXPORT(generate_grain_y_16bpc);
HWY_EXPORT(generate_grain_uv_420_16bpc);
HWY_EXPORT(generate_grain_uv_422_16bpc);
HWY_EXPORT(generate_grain_uv_444_16bpc);
HWY_EXPORT(fgy_32x32xn_16bpc);
HWY_EXPORT(fguv_32x32xn_420_16bpc);
HWY_EXPORT(fguv_32x32xn_422_16bpc);
HWY_EXPORT(fguv_32x32xn_444_16bpc);
}  // namespace dav1d

namespace {
// Mirrors of Dav1dFilmGrainDSPContext (src/filmgrain.h), so that this file
// does not need dav1d's bitdepth-templated C headers.
using FgGenY8 = void (*)(int8_t (*)[82], const Dav1dFilmGrainData *);
using FgGenUV8 = void (*)(int8_t (*)[82], const int8_t (*)[82],
                          const Dav1dFilmGrainData *, intptr_t);
using FgyFn8 = void (*)(uint8_t *, const uint8_t *, ptrdiff_t,
                        const Dav1dFilmGrainData *, size_t, const uint8_t *,
                        const int8_t (*)[82], int, int);
using FguvFn8 = void (*)(uint8_t *, const uint8_t *, ptrdiff_t,
                         const Dav1dFilmGrainData *, size_t, const uint8_t *,
                         const int8_t (*)[82], int, int, const uint8_t *,
                         ptrdiff_t, int, int);
struct FgDSP8 {
    FgGenY8 generate_grain_y;
    FgGenUV8 generate_grain_uv[3];
    FgyFn8 fgy_32x32xn;
    FguvFn8 fguv_32x32xn[3];
};

using FgGenY16 = void (*)(int16_t (*)[82], const Dav1dFilmGrainData *, int);
using FgGenUV16 = void (*)(int16_t (*)[82], const int16_t (*)[82],
                           const Dav1dFilmGrainData *, intptr_t, int);
using FgyFn16 = void (*)(uint16_t *, const uint16_t *, ptrdiff_t,
                         const Dav1dFilmGrainData *, size_t, const uint8_t *,
                         const int16_t (*)[82], int, int, int);
using FguvFn16 = void (*)(uint16_t *, const uint16_t *, ptrdiff_t,
                          const Dav1dFilmGrainData *, size_t, const uint8_t *,
                          const int16_t (*)[82], int, int, const uint16_t *,
                          ptrdiff_t, int, int, int);
struct FgDSP16 {
    FgGenY16 generate_grain_y;
    FgGenUV16 generate_grain_uv[3];
    FgyFn16 fgy_32x32xn;
    FguvFn16 fguv_32x32xn[3];
};
}  // namespace

namespace dav1d {

// Resolve the best per-target function pointers once at init; the
// ChosenTarget must be initialized first or the tables yield their
// re-dispatching first entry.
static void hwy_init_chosen_target() {
    hwy::GetChosenTarget().Update(hwy::SupportedTargets());
}

static void film_grain_dsp_init_8bpc_hwy(void *const c) {
    auto *const ctx = static_cast<FgDSP8 *>(c);
    ctx->generate_grain_y = HWY_DYNAMIC_POINTER(generate_grain_y_8bpc);
    ctx->generate_grain_uv[0] = HWY_DYNAMIC_POINTER(generate_grain_uv_420_8bpc);
    ctx->generate_grain_uv[1] = HWY_DYNAMIC_POINTER(generate_grain_uv_422_8bpc);
    ctx->generate_grain_uv[2] = HWY_DYNAMIC_POINTER(generate_grain_uv_444_8bpc);
    ctx->fgy_32x32xn = HWY_DYNAMIC_POINTER(fgy_32x32xn_8bpc);
    ctx->fguv_32x32xn[0] = HWY_DYNAMIC_POINTER(fguv_32x32xn_420_8bpc);
    ctx->fguv_32x32xn[1] = HWY_DYNAMIC_POINTER(fguv_32x32xn_422_8bpc);
    ctx->fguv_32x32xn[2] = HWY_DYNAMIC_POINTER(fguv_32x32xn_444_8bpc);
}

static void film_grain_dsp_init_16bpc_hwy(void *const c) {
    auto *const ctx = static_cast<FgDSP16 *>(c);
    ctx->generate_grain_y = HWY_DYNAMIC_POINTER(generate_grain_y_16bpc);
    ctx->generate_grain_uv[0] = HWY_DYNAMIC_POINTER(generate_grain_uv_420_16bpc);
    ctx->generate_grain_uv[1] = HWY_DYNAMIC_POINTER(generate_grain_uv_422_16bpc);
    ctx->generate_grain_uv[2] = HWY_DYNAMIC_POINTER(generate_grain_uv_444_16bpc);
    ctx->fgy_32x32xn = HWY_DYNAMIC_POINTER(fgy_32x32xn_16bpc);
    ctx->fguv_32x32xn[0] = HWY_DYNAMIC_POINTER(fguv_32x32xn_420_16bpc);
    ctx->fguv_32x32xn[1] = HWY_DYNAMIC_POINTER(fguv_32x32xn_422_16bpc);
    ctx->fguv_32x32xn[2] = HWY_DYNAMIC_POINTER(fguv_32x32xn_444_16bpc);
}

}  // namespace dav1d

extern "C" void dav1d_film_grain_dsp_init_hwy_8bpc(void *const c) {
    dav1d::hwy_init_chosen_target();
    dav1d::film_grain_dsp_init_8bpc_hwy(c);
}

extern "C" void dav1d_film_grain_dsp_init_hwy_16bpc(void *const c) {
    dav1d::hwy_init_chosen_target();
    dav1d::film_grain_dsp_init_16bpc_hwy(c);
}

#endif  // HWY_ONCE
