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

// Motion compensation compound/utility kernels (src/mc_tmpl.c: avg, w_avg,
// mask, w_mask, blend, blend_v, blend_h, emu_edge, resize) implemented with
// Google Highway: one source is compiled per SIMD target and the best one
// supported by the CPU is selected at runtime (HWY_DYNAMIC_DISPATCH).
// Bit-exact with the C code; warp8x8 and the scaled mc variants are not
// covered.
//
// Vectors are capped at 128 bits. All widening
// is done on sequential half vectors (PromoteTo of Lower/UpperHalf) rather
// than MulEven/MulOdd, so no deinterleaving/reinterleaving shuffles are
// needed; the i32 -> i16 pack is a pair of plain narrowing stores.
//
// Value-range notes: the i16 prep inputs are in [-5132, 9212] (8bpc) resp.
// [-28794, 28791] (12bpc), see PREP_BIAS in src/mc_tmpl.c; every i32 dot
// below stays below 2^22 in magnitude, and every post-shift value fits i16.
// Where an i32 -> i16 demote could still saturate (out-of-contract inputs),
// the final clamp to [0, bitdepth_max] (<= 4095) makes it equivalent to the
// C code's iclip_pixel from int.

#include <stddef.h>
#include <stdint.h>
#include <type_traits>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/hwy/mc_compound.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "src/hwy/common.h"

// Defined in src/tables.c.
extern "C" const uint8_t dav1d_obmc_masks[64];
extern "C" const int8_t dav1d_resize_filter[64][8];

HWY_BEFORE_NAMESPACE();

namespace dav1d {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// intermediate_bits/PREP_BIAS of src/mc_tmpl.c: the prep output is scaled by
// 2^intermediate_bits and biased so that it fits int16.
template <typename Pixel>
static HWY_INLINE void mc_intermediate(const int bitdepth_max, int& ib,
                                       int& bias) {
    if (sizeof(Pixel) == 1) {
        ib = 4;
        bias = 0;
    } else {
        ib = 13 - hwy_ulog2((unsigned) bitdepth_max);
        bias = 8192;
    }
}

// ((dot + rnd) >> sh) clipped to [0, bitdepth_max], stored as pixels.
template <int kCap, class D16, typename Pixel, class V32>
static HWY_INLINE void StorePackPx(const D16 d, Pixel *const p, const size_t n,
                                   V32 lo, V32 hi, const int rnd, const int sh,
                                   const int bitdepth_max) {
    const hn::DFromV<V32> d32;
    const auto vr = hn::Set(d32, rnd);
    lo = hn::ShiftRightSame(hn::Add(lo, vr), sh);
    hi = hn::ShiftRightSame(hn::Add(hi, vr), sh);
    const hn::Rebind<Pixel, D16> dp;
    if constexpr (sizeof(Pixel) == 1) {
        // The saturating u8 demote is the clip to [0, 255].
        const auto v = PackHalves(d, lo, hi);
        if (kCap) hn::StoreU(hn::DemoteTo(dp, v), dp, p);
        else hn::StoreN(hn::DemoteTo(dp, v), dp, p, n);
    } else {
        const auto v = hn::Clamp(PackHalves(d, lo, hi), hn::Zero(d),
                                 hn::Set(d, bitdepth_max));
        if (kCap) hn::StoreU(hn::BitCast(dp, v), dp, p);
        else hn::StoreN(hn::BitCast(dp, v), dp, p, n);
    }
}

// avg_c from src/mc_tmpl.c.
template <typename Pixel, int kCap>
static void hwy_avg_impl(Pixel *dst, const ptrdiff_t dst_stride,
                         const int16_t *tmp1, const int16_t *tmp2,
                         const int w, const int h, const int bitdepth_max,
                         const std::integral_constant<int, kCap>) {
    int ib, bias;
    mc_intermediate<Pixel>(bitdepth_max, ib, bias);
    const int rnd = (1 << ib) + 2 * bias, sh = ib + 1;
    const ptrdiff_t ds = dst_stride / (ptrdiff_t) sizeof(Pixel);
    const hn::CappedTag<int16_t, kCap == 0 ? 8 : kCap> d;
    const hn::Repartition<int32_t, decltype(d)> d32;
    const int N = (int) hn::Lanes(d);
    const auto vrnd = hn::Set(d, rnd);
    for (int y = 0; y < h; y++, tmp1 += w, tmp2 += w, dst += ds)
        for (int x = 0; x < w; x += N) {
            const size_t n = (size_t) (w - x < N ? w - x : N);
            const auto t1 = LoadI16<kCap>(d, tmp1 + x, n);
            const auto t2 = LoadI16<kCap>(d, tmp2 + x, n);
            if constexpr (sizeof(Pixel) == 1) {
                // tmp1 + tmp2 + 16 fits i16 (inputs are in [-5132, 9212]), so
                // no widening; the saturating u8 demote in StorePx is the
                // clip (values are in [-321, 576]).
                const auto v = hn::ShiftRightSame(
                    hn::Add(hn::Add(t1, t2), vrnd), sh);
                StorePx<kCap>(d, dst + x, n, v);
            } else {
                const auto lo = hn::Add(WidenLo(d32, t1), WidenLo(d32, t2));
                const auto hi = hn::Add(WidenHi(d32, t1), WidenHi(d32, t2));
                StorePackPx<kCap>(d, dst + x, n, lo, hi, rnd, sh,
                                  bitdepth_max);
            }
        }
}

// w_avg_c from src/mc_tmpl.c.
template <typename Pixel, int kCap>
static void hwy_w_avg_impl(Pixel *dst, const ptrdiff_t dst_stride,
                           const int16_t *tmp1, const int16_t *tmp2,
                           const int w, const int h, const int weight,
                           const int bitdepth_max,
                           const std::integral_constant<int, kCap>) {
    int ib, bias;
    mc_intermediate<Pixel>(bitdepth_max, ib, bias);
    const int rnd = (8 << ib) + 16 * bias, sh = ib + 4;
    const ptrdiff_t ds = dst_stride / (ptrdiff_t) sizeof(Pixel);
    const hn::CappedTag<int16_t, kCap == 0 ? 8 : kCap> d;
    const hn::Repartition<int32_t, decltype(d)> d32;
    const int N = (int) hn::Lanes(d);
    // Weights widened once; the products then lower to widening muls.
    const auto w1 = hn::Set(d, weight);
    const auto w2 = hn::Set(d, 16 - weight);
    const auto w1_lo = WidenLo(d32, w1), w1_hi = WidenHi(d32, w1);
    const auto w2_lo = WidenLo(d32, w2), w2_hi = WidenHi(d32, w2);
    for (int y = 0; y < h; y++, tmp1 += w, tmp2 += w, dst += ds)
        for (int x = 0; x < w; x += N) {
            const size_t n = (size_t) (w - x < N ? w - x : N);
            const auto t1 = LoadI16<kCap>(d, tmp1 + x, n);
            const auto t2 = LoadI16<kCap>(d, tmp2 + x, n);
            const auto lo = hn::Add(hn::Mul(WidenLo(d32, t1), w1_lo),
                                    hn::Mul(WidenLo(d32, t2), w2_lo));
            const auto hi = hn::Add(hn::Mul(WidenHi(d32, t1), w1_hi),
                                    hn::Mul(WidenHi(d32, t2), w2_hi));
            StorePackPx<kCap>(d, dst + x, n, lo, hi, rnd, sh, bitdepth_max);
        }
}

// mask_c from src/mc_tmpl.c.
template <typename Pixel, int kCap>
static void hwy_mask_impl(Pixel *dst, const ptrdiff_t dst_stride,
                          const int16_t *tmp1, const int16_t *tmp2,
                          const int w, const int h, const uint8_t *mask,
                          const int bitdepth_max,
                          const std::integral_constant<int, kCap>) {
    int ib, bias;
    mc_intermediate<Pixel>(bitdepth_max, ib, bias);
    const int rnd = (32 << ib) + 64 * bias, sh = ib + 6;
    const ptrdiff_t ds = dst_stride / (ptrdiff_t) sizeof(Pixel);
    const hn::CappedTag<int16_t, kCap == 0 ? 8 : kCap> d;
    const hn::Repartition<int32_t, decltype(d)> d32;
    const int N = (int) hn::Lanes(d);
    const auto v64 = hn::Set(d, 64);
    for (int y = 0; y < h; y++, tmp1 += w, tmp2 += w, mask += w, dst += ds)
        for (int x = 0; x < w; x += N) {
            const size_t n = (size_t) (w - x < N ? w - x : N);
            const auto t1 = LoadI16<kCap>(d, tmp1 + x, n);
            const auto t2 = LoadI16<kCap>(d, tmp2 + x, n);
            const auto m = LoadI16<kCap>(d, mask + x, n);
            const auto m64 = hn::Sub(v64, m);
            const auto lo = hn::Add(hn::Mul(WidenLo(d32, t1), WidenLo(d32, m)),
                                    hn::Mul(WidenLo(d32, t2),
                                            WidenLo(d32, m64)));
            const auto hi = hn::Add(hn::Mul(WidenHi(d32, t1), WidenHi(d32, m)),
                                    hn::Mul(WidenHi(d32, t2),
                                            WidenHi(d32, m64)));
            StorePackPx<kCap>(d, dst + x, n, lo, hi, rnd, sh, bitdepth_max);
        }
}

// w_mask_c from src/mc_tmpl.c, specialized for 444/422/420. The per-pixel
// mask m = imin(38 + ((|tmp1 - tmp2| + mask_rnd) >> mask_sh), 64) is in
// [38, 64]. With ss_hor the mask is stored at half horizontal resolution as
// pair sums m[x] + m[x+1] (extracted from the packed i16 lanes via the
// 32-bit view); with ss_ver even rows store the raw pair sum (m + n <= 128,
// fits u8) and odd rows fold it back in: (m + n + prev + 2 - sign) >> 2.
template <typename Pixel, int kCap, bool kSsHor, bool kSsVer>
static void hwy_w_mask_impl(Pixel *dst, const ptrdiff_t dst_stride,
                            const int16_t *tmp1, const int16_t *tmp2,
                            const int w, int h, uint8_t *mask, const int sign,
                            const int bitdepth_max,
                            const std::integral_constant<int, kCap>) {
    int ib, bias;
    mc_intermediate<Pixel>(bitdepth_max, ib, bias);
    const int bitdepth = sizeof(Pixel) == 1 ? 8 :
        hwy_ulog2((unsigned) bitdepth_max) + 1;
    const int rnd = (32 << ib) + 64 * bias, sh = ib + 6;
    const int mask_sh = bitdepth + ib - 4, mask_rnd = 1 << (mask_sh - 5);
    const ptrdiff_t ds = dst_stride / (ptrdiff_t) sizeof(Pixel);
    const hn::CappedTag<int16_t, kCap == 0 ? 8 : kCap> d;
    const hn::Repartition<int32_t, decltype(d)> d32;
    const int N = (int) hn::Lanes(d);
    const auto vmr16 = hn::Set(d, mask_rnd);
    const auto v38_16 = hn::Set(d, 38);
    const auto v64_16 = hn::Set(d, 64);
    do {
        for (int x = 0; x < w; x += N) {
            const size_t n = (size_t) (w - x < N ? w - x : N);
            const auto t1 = LoadI16<kCap>(d, tmp1 + x, n);
            const auto t2 = LoadI16<kCap>(d, tmp2 + x, n);
            // |tmp1 - tmp2| needs 17 bits (16bpc), but m already saturates at
            // 64 for |diff| >= 26592, so the i16 saturated diff yields the
            // exact m for every input: max(sat(a-b), sat(b-a)) is in
            // [0, 32767], and the SaturatedAdd of mask_rnd can again only
            // saturate where the result clamps to 64.
            const auto adiff = hn::Max(hn::SaturatedSub(t1, t2),
                                       hn::SaturatedSub(t2, t1));
            const auto m16 = hn::Min(hn::Add(hn::ShiftRightSame(
                hn::SaturatedAdd(adiff, vmr16), mask_sh), v38_16), v64_16);
            const auto m64 = hn::Sub(v64_16, m16);
            // dst = (tmp1 * m + tmp2 * (64 - m) + rnd) >> sh, clipped.
            const auto r_lo = hn::Add(hn::Mul(WidenLo(d32, t1),
                                              WidenLo(d32, m16)),
                                      hn::Mul(WidenLo(d32, t2),
                                              WidenLo(d32, m64)));
            const auto r_hi = hn::Add(hn::Mul(WidenHi(d32, t1),
                                              WidenHi(d32, m16)),
                                      hn::Mul(WidenHi(d32, t2),
                                              WidenHi(d32, m64)));
            StorePackPx<kCap>(d, dst + x, n, r_lo, r_hi, rnd, sh,
                              bitdepth_max);
            if constexpr (!kSsHor) {
                const hn::Rebind<uint8_t, decltype(d)> d8;
                if (kCap) hn::StoreU(hn::DemoteTo(d8, m16), d8, mask + x);
                else hn::StoreN(hn::DemoteTo(d8, m16), d8, mask + x, n);
            } else {
                // Reinterpret adjacent i16 lanes as one i32 lane: the pair
                // sum is low half + high half; linear half-resolution order.
                const hn::Rebind<uint8_t, decltype(d32)> d8h;
                const auto u = hn::BitCast(d32, m16);
                const auto pairs = hn::Add(hn::And(u, hn::Set(d32, 0xffff)),
                                           hn::ShiftRight<16>(u));
                if constexpr (!kSsVer) {
                    const auto q = hn::ShiftRightSame(
                        hn::Add(pairs, hn::Set(d32, 1 - sign)), 1);
                    const auto q8 = hn::DemoteTo(d8h, q);
                    if (kCap) hn::StoreU(q8, d8h, mask + (x >> 1));
                    else hn::StoreN(q8, d8h, mask + (x >> 1), n >> 1);
                } else if (h & 1) {
                    const hn::Rebind<int16_t, decltype(d32)> d16h;
                    const auto prev = hn::PromoteTo(d32,
                        LoadI16<0>(d16h, mask + (x >> 1), n >> 1));
                    const auto q = hn::ShiftRightSame(hn::Add(
                        hn::Add(pairs, prev), hn::Set(d32, 2 - sign)), 2);
                    const auto q8 = hn::DemoteTo(d8h, q);
                    if (kCap) hn::StoreU(q8, d8h, mask + (x >> 1));
                    else hn::StoreN(q8, d8h, mask + (x >> 1), n >> 1);
                } else {
                    const auto p8 = hn::DemoteTo(d8h, pairs);
                    if (kCap) hn::StoreU(p8, d8h, mask + (x >> 1));
                    else hn::StoreN(p8, d8h, mask + (x >> 1), n >> 1);
                }
            }
        }
        tmp1 += w;
        tmp2 += w;
        dst += ds;
        // Same row parity as the C code: the mask row advances on odd
        // remaining-height rows for ss_ver.
        if (!kSsVer || (h & 1)) mask += w >> kSsHor;
    } while (--h);
}

// blend_px from src/mc_tmpl.c: ((a * (64 - m) + b * m) + 32) >> 6 with no
// clip; a, b are valid pixels and m in [0, 64], so the result is already
// within [0, bitdepth_max]. 8bpc works on whole u8 vectors: every
// intermediate <= 255 * 64 + 32 = 16352 fits u16, so the zext/multiplies are
// exact and 64 - m is exact in u8.
template <class D8, class V8>
static HWY_INLINE hn::VFromD<D8> BlendPx8(const D8 d8, const V8 a, const V8 b,
                                          const V8 m) {
    const hn::Repartition<uint16_t, D8> d16;
    const auto m64 = hn::Sub(hn::Set(d8, 64), m);
    const auto lo = hn::Add(hn::Mul(hn::PromoteTo(d16, hn::LowerHalf(a)),
                                    hn::PromoteTo(d16, hn::LowerHalf(m64))),
                            hn::Mul(hn::PromoteTo(d16, hn::LowerHalf(b)),
                                    hn::PromoteTo(d16, hn::LowerHalf(m))));
    const auto hi = hn::Add(hn::Mul(hn::PromoteUpperTo(d16, a),
                                    hn::PromoteUpperTo(d16, m64)),
                            hn::Mul(hn::PromoteUpperTo(d16, b),
                                    hn::PromoteUpperTo(d16, m)));
    const hn::Rebind<uint8_t, decltype(d16)> d8h;
    const auto vr = hn::Set(d16, 32);
    // Non-saturating narrow: the result is <= 255, so truncation is exact.
    return hn::Combine(d8,
        hn::TruncateTo(d8h, hn::ShiftRightSame(hn::Add(hi, vr), 6)),
        hn::TruncateTo(d8h, hn::ShiftRightSame(hn::Add(lo, vr), 6)));
}

// 16bpc variant on i16 lanes: the products need i32 (max 4095 * 64 + 32 =
// 262112); the i32 -> i16 demote is exact (result <= bitdepth_max <= 4095).
template <int kCap, class D16, typename Pixel>
static HWY_INLINE hn::VFromD<D16> BlendPx16(const D16 d, const Pixel *const a,
                                            const Pixel *const b,
                                            const size_t n,
                                            const hn::VFromD<D16> m) {
    const hn::Repartition<int32_t, D16> d32;
    const auto va = LoadI16<kCap>(d, a, n);
    const auto vb = LoadI16<kCap>(d, b, n);
    const auto m64 = hn::Sub(hn::Set(d, 64), m);
    const auto vr = hn::Set(d32, 32);
    const auto lo = hn::ShiftRightSame(hn::Add(hn::Add(
        hn::Mul(WidenLo(d32, va), WidenLo(d32, m64)),
        hn::Mul(WidenLo(d32, vb), WidenLo(d32, m))), vr), 6);
    const auto hi = hn::ShiftRightSame(hn::Add(hn::Add(
        hn::Mul(WidenHi(d32, va), WidenHi(d32, m64)),
        hn::Mul(WidenHi(d32, vb), WidenHi(d32, m))), vr), 6);
    return PackHalves(d, lo, hi);
}

// blend_c from src/mc_tmpl.c.
template <typename Pixel, int kCap>
static void hwy_blend_impl(Pixel *dst, const ptrdiff_t dst_stride,
                           const Pixel *tmp, const int w, int h,
                           const uint8_t *mask,
                           const std::integral_constant<int, kCap>) {
    const ptrdiff_t ds = dst_stride / (ptrdiff_t) sizeof(Pixel);
    if constexpr (sizeof(Pixel) == 1) {
        const hn::ScalableTag<Pixel> d8;
        const int N = (int) hn::Lanes(d8);
        for (int y = 0; y < h; y++, tmp += w, mask += w, dst += ds) {
            int x = 0;
            for (; x + 2 * N <= w; x += 2 * N) {
                const auto v0 = BlendPx8(d8, hn::LoadU(d8, dst + x),
                                         hn::LoadU(d8, tmp + x),
                                         hn::LoadU(d8, mask + x));
                const auto v1 = BlendPx8(d8, hn::LoadU(d8, dst + x + N),
                                         hn::LoadU(d8, tmp + x + N),
                                         hn::LoadU(d8, mask + x + N));
                hn::StoreU(v0, d8, dst + x);
                hn::StoreU(v1, d8, dst + x + N);
            }
            for (; x + N <= w; x += N)
                hn::StoreU(BlendPx8(d8, hn::LoadU(d8, dst + x),
                                    hn::LoadU(d8, tmp + x),
                                    hn::LoadU(d8, mask + x)), d8, dst + x);
            if (x < w) {
                const size_t n = (size_t) (w - x);
                hn::StoreN(BlendPx8(d8, hn::LoadN(d8, dst + x, n),
                                    hn::LoadN(d8, tmp + x, n),
                                    hn::LoadN(d8, mask + x, n)), d8, dst + x,
                           n);
            }
        }
    } else {
        const hn::CappedTag<int16_t, kCap == 0 ? 8 : kCap> d;
        const int N = (int) hn::Lanes(d);
        for (int y = 0; y < h; y++, tmp += w, mask += w, dst += ds)
            for (int x = 0; x < w; x += N) {
                const size_t n = (size_t) (w - x < N ? w - x : N);
                const auto m = LoadI16<kCap>(d, mask + x, n);
                StorePx<kCap>(d, dst + x, n,
                              BlendPx16<kCap>(d, dst + x, tmp + x, n, m));
            }
    }
}

// blend_v_c from src/mc_tmpl.c: blends only the left (w * 3) >> 2 columns.
template <typename Pixel, int kCap>
static void hwy_blend_v_impl(Pixel *dst, const ptrdiff_t dst_stride,
                             const Pixel *tmp, const int w, int h,
                             const std::integral_constant<int, kCap>) {
    const uint8_t *const mask = &dav1d_obmc_masks[w];
    const ptrdiff_t ds = dst_stride / (ptrdiff_t) sizeof(Pixel);
    const int cols = (w * 3) >> 2;
    // Column chunks in the outer loop: the mask row is loaded once per chunk.
    if constexpr (sizeof(Pixel) == 1) {
        const hn::ScalableTag<Pixel> d8;
        const int N = (int) hn::Lanes(d8);
        int x = 0;
        for (; x + N <= cols; x += N) {
            const auto m = hn::LoadU(d8, mask + x);
            const Pixel *t = tmp + x;
            Pixel *dd = dst + x;
            for (int y = 0; y < h; y++, t += w, dd += ds)
                hn::StoreU(BlendPx8(d8, hn::LoadU(d8, dd), hn::LoadU(d8, t),
                                    m), d8, dd);
        }
        if (x < cols) {
            const size_t n = (size_t) (cols - x);
            const auto m = hn::LoadN(d8, mask + x, n);
            const Pixel *t = tmp + x;
            Pixel *dd = dst + x;
            for (int y = 0; y < h; y++, t += w, dd += ds)
                hn::StoreN(BlendPx8(d8, hn::LoadN(d8, dd, n),
                                    hn::LoadN(d8, t, n), m), d8, dd, n);
        }
    } else {
        const hn::CappedTag<int16_t, kCap == 0 ? 8 : kCap> d;
        const int N = (int) hn::Lanes(d);
        int x = 0;
        for (; x + N <= cols; x += N) {
            const auto m = LoadI16<8>(d, mask + x, N);
            const Pixel *t = tmp + x;
            Pixel *dd = dst + x;
            for (int y = 0; y < h; y++, t += w, dd += ds)
                StorePx<8>(d, dd, N, BlendPx16<8>(d, dd, t, N, m));
        }
        if (x < cols) {
            const size_t n = (size_t) (cols - x);
            const auto m = LoadI16<0>(d, mask + x, n);
            const Pixel *t = tmp + x;
            Pixel *dd = dst + x;
            for (int y = 0; y < h; y++, t += w, dd += ds)
                StorePx<0>(d, dd, n, BlendPx16<0>(d, dd, t, n, m));
        }
    }
}

// blend_h_c from src/mc_tmpl.c: one mask value per row, (h * 3) >> 2 rows.
template <typename Pixel, int kCap>
static void hwy_blend_h_impl(Pixel *dst, const ptrdiff_t dst_stride,
                             const Pixel *tmp, const int w, const int h,
                             const std::integral_constant<int, kCap>) {
    const uint8_t *mask = &dav1d_obmc_masks[h];
    const ptrdiff_t ds = dst_stride / (ptrdiff_t) sizeof(Pixel);
    int rows = (h * 3) >> 2;
    if constexpr (sizeof(Pixel) == 1) {
        const hn::ScalableTag<Pixel> d8;
        const int N = (int) hn::Lanes(d8);
        do {
            const auto m = hn::Set(d8, *mask++);
            int x = 0;
            for (; x + 2 * N <= w; x += 2 * N) {
                const auto v0 = BlendPx8(d8, hn::LoadU(d8, dst + x),
                                         hn::LoadU(d8, tmp + x), m);
                const auto v1 = BlendPx8(d8, hn::LoadU(d8, dst + x + N),
                                         hn::LoadU(d8, tmp + x + N), m);
                hn::StoreU(v0, d8, dst + x);
                hn::StoreU(v1, d8, dst + x + N);
            }
            for (; x + N <= w; x += N)
                hn::StoreU(BlendPx8(d8, hn::LoadU(d8, dst + x),
                                    hn::LoadU(d8, tmp + x), m), d8, dst + x);
            if (x < w) {
                const size_t n = (size_t) (w - x);
                hn::StoreN(BlendPx8(d8, hn::LoadN(d8, dst + x, n),
                                    hn::LoadN(d8, tmp + x, n), m), d8,
                           dst + x, n);
            }
            tmp += w;
            dst += ds;
        } while (--rows);
    } else {
        const hn::CappedTag<int16_t, kCap == 0 ? 8 : kCap> d;
        const int N = (int) hn::Lanes(d);
        do {
            const auto m = hn::Set(d, *mask++);
            for (int x = 0; x < w; x += N) {
                const size_t n = (size_t) (w - x < N ? w - x : N);
                StorePx<kCap>(d, dst + x, n,
                              BlendPx16<kCap>(d, dst + x, tmp + x, n, m));
            }
            tmp += w;
            dst += ds;
        } while (--rows);
    }
}

// emu_edge_c from src/mc_tmpl.c: blits the visible portion and extends the
// edges; pure pixel copies. The row copies/set use overlapping full-vector
// tails within the same row (identical values), never touching other rows.
template <typename Pixel>
static void hwy_emu_edge(const intptr_t bw, const intptr_t bh,
                         const intptr_t iw, const intptr_t ih,
                         const intptr_t x, const intptr_t y,
                         Pixel *const dst, const ptrdiff_t dst_stride,
                         const Pixel *ref, const ptrdiff_t ref_stride) {
    const hn::ScalableTag<Pixel> d; // copy-only: no lane-order concerns
    const int N = (int) hn::Lanes(d);
    const ptrdiff_t ds = dst_stride / (ptrdiff_t) sizeof(Pixel);
    const ptrdiff_t rs = ref_stride / (ptrdiff_t) sizeof(Pixel);
    const auto clip = [](const int v, const int hi) {
        return v < 0 ? 0 : v > hi ? hi : v;
    };
    const auto copy = [&](Pixel *const dp, const Pixel *const s,
                          const int count) {
        if (count >= N) {
            int i = 0;
            for (; i + 4 * N <= count; i += 4 * N) {
                const auto v0 = hn::LoadU(d, s + i);
                const auto v1 = hn::LoadU(d, s + i + N);
                const auto v2 = hn::LoadU(d, s + i + 2 * N);
                const auto v3 = hn::LoadU(d, s + i + 3 * N);
                hn::StoreU(v0, d, dp + i);
                hn::StoreU(v1, d, dp + i + N);
                hn::StoreU(v2, d, dp + i + 2 * N);
                hn::StoreU(v3, d, dp + i + 3 * N);
            }
            for (; i + N <= count; i += N)
                hn::StoreU(hn::LoadU(d, s + i), d, dp + i);
            if (i < count)
                hn::StoreU(hn::LoadU(d, s + count - N), d, dp + count - N);
        } else {
            hn::StoreN(hn::LoadN(d, s, (size_t) count), d, dp,
                       (size_t) count);
        }
    };
    const auto set = [&](Pixel *const dp, const Pixel v, const int count) {
        const auto vv = hn::Set(d, v);
        if (count >= N) {
            int i = 0;
            for (; i + N <= count; i += N) hn::StoreU(vv, d, dp + i);
            if (i < count) hn::StoreU(vv, d, dp + count - N);
        } else {
            hn::StoreN(vv, d, dp, (size_t) count);
        }
    };

    ref += clip((int) y, (int) ih - 1) * rs + clip((int) x, (int) iw - 1);
    const int left = clip((int) -x, (int) bw - 1);
    const int right = clip((int) (x + bw - iw), (int) bw - 1);
    const int top = clip((int) -y, (int) bh - 1);
    const int bottom = clip((int) (y + bh - ih), (int) bh - 1);
    const int cw = (int) bw - left - right;
    const int ch = (int) bh - top - bottom;

    Pixel *blk = dst + top * ds;
    if (cw >= N && left < N && right < N) {
        // Fast path: store the edge values as full vectors overlapping the
        // center, then copy the center over the overlap. All stores stay
        // within the row and the final values are exact.
        for (int yy = 0; yy < ch; yy++, ref += rs, blk += ds) {
            if (left) hn::StoreU(hn::Set(d, ref[0]), d, blk);
            if (right) hn::StoreU(hn::Set(d, ref[cw - 1]), d, blk + bw - N);
            copy(blk + left, ref, cw);
        }
    } else {
        for (int yy = 0; yy < ch; yy++, ref += rs, blk += ds) {
            copy(blk + left, ref, cw);
            if (left) set(blk, blk[left], left);
            if (right) set(blk + left + cw, blk[left + cw - 1], right);
        }
    }

    const Pixel *const first = dst + top * ds;
    Pixel *row = dst;
    for (int yy = 0; yy < top; yy++, row += ds) copy(row, first, (int) bw);

    row = dst + (top + ch) * ds;
    for (int yy = 0; yy < bottom; yy++, row += ds)
        copy(row, row - ds, (int) bw);
}

// resize_c from src/mc_tmpl.c: per output pixel an 8-tap dot over a window
// clipped to [0, src_w). The clip can only trigger near the borders, so the
// interior loads the window directly. |dot| <= 8 * 128 * 4095 fits i32 (the
// taps are int8, pixels <= 4095).
template <typename Pixel>
static void hwy_resize(Pixel *dst, const ptrdiff_t dst_stride,
                       const Pixel *src, const ptrdiff_t src_stride,
                       const int dst_w, int h, const int src_w, const int dx,
                       const int mx0, const int bitdepth_max) {
    const hn::FixedTag<int16_t, 8> d16;
    const hn::Repartition<int32_t, decltype(d16)> d32;
    const hn::FixedTag<int8_t, 8> d8;
    const ptrdiff_t ds = dst_stride / (ptrdiff_t) sizeof(Pixel);
    const ptrdiff_t ss = src_stride / (ptrdiff_t) sizeof(Pixel);
    do {
        int mx = mx0, src_x = -1;
        for (int x = 0; x < dst_w; x++) {
            const int8_t *const F = dav1d_resize_filter[mx >> 8];
            alignas(16) Pixel win[8];
            const Pixel *wp;
            if (src_x >= 3 && src_x + 4 < src_w) {
                wp = src + (src_x - 3);
            } else {
                for (int k = 0; k < 8; k++) {
                    const int idx = src_x - 3 + k;
                    win[k] = src[idx < 0 ? 0 :
                                 idx >= src_w ? src_w - 1 : idx];
                }
                wp = win;
            }
            const auto p16 = LoadI16<8>(d16, wp, 8);
            const auto f16 = hn::PromoteTo(d16, hn::LoadU(d8, F));
            const auto dot = hn::Add(hn::MulEven(p16, f16),
                                     hn::MulOdd(p16, f16));
            const int s = (int) hn::ReduceSum(d32, dot);
            const int v = (64 - s) >> 7;
            dst[x] = (Pixel) (v < 0 ? 0 : v > bitdepth_max ? bitdepth_max : v);
            mx += dx;
            src_x += mx >> 14;
            mx &= 0x3fff;
        }
        dst += ds;
        src += ss;
    } while (--h);
}

// Widths >= 8 are multiples of 8 in dav1d, so the partial-chunk path only
// ever sees w < 8.
template <typename Pixel>
static HWY_INLINE void hwy_avg(Pixel *const dst, const ptrdiff_t dst_stride,
                               const int16_t *const tmp1,
                               const int16_t *const tmp2, const int w,
                               const int h, const int bitdepth_max) {
    if (w % 8 == 0)
        hwy_avg_impl(dst, dst_stride, tmp1, tmp2, w, h, bitdepth_max,
                     std::integral_constant<int, 8>{});
    else
        hwy_avg_impl(dst, dst_stride, tmp1, tmp2, w, h, bitdepth_max,
                     std::integral_constant<int, 0>{});
}

template <typename Pixel>
static HWY_INLINE void hwy_w_avg(Pixel *const dst, const ptrdiff_t dst_stride,
                                 const int16_t *const tmp1,
                                 const int16_t *const tmp2, const int w,
                                 const int h, const int weight,
                                 const int bitdepth_max) {
    if (w % 8 == 0)
        hwy_w_avg_impl(dst, dst_stride, tmp1, tmp2, w, h, weight, bitdepth_max,
                       std::integral_constant<int, 8>{});
    else
        hwy_w_avg_impl(dst, dst_stride, tmp1, tmp2, w, h, weight, bitdepth_max,
                       std::integral_constant<int, 0>{});
}

template <typename Pixel>
static HWY_INLINE void hwy_mask(Pixel *const dst, const ptrdiff_t dst_stride,
                                const int16_t *const tmp1,
                                const int16_t *const tmp2, const int w,
                                const int h, const uint8_t *const mask,
                                const int bitdepth_max) {
    if (w % 8 == 0)
        hwy_mask_impl(dst, dst_stride, tmp1, tmp2, w, h, mask, bitdepth_max,
                      std::integral_constant<int, 8>{});
    else
        hwy_mask_impl(dst, dst_stride, tmp1, tmp2, w, h, mask, bitdepth_max,
                      std::integral_constant<int, 0>{});
}

template <typename Pixel, bool kSsHor, bool kSsVer>
static HWY_INLINE void hwy_w_mask(Pixel *const dst, const ptrdiff_t dst_stride,
                                  const int16_t *const tmp1,
                                  const int16_t *const tmp2, const int w,
                                  const int h, uint8_t *const mask,
                                  const int sign, const int bitdepth_max) {
    if (w % 8 == 0)
        hwy_w_mask_impl<Pixel, 8, kSsHor, kSsVer>(dst, dst_stride, tmp1, tmp2,
                                                  w, h, mask, sign,
                                                  bitdepth_max,
                                                  std::integral_constant<int, 8>{});
    else
        hwy_w_mask_impl<Pixel, 0, kSsHor, kSsVer>(dst, dst_stride, tmp1, tmp2,
                                                  w, h, mask, sign,
                                                  bitdepth_max,
                                                  std::integral_constant<int, 0>{});
}

template <typename Pixel>
static HWY_INLINE void hwy_blend(Pixel *const dst, const ptrdiff_t dst_stride,
                                 const Pixel *const tmp, const int w,
                                 const int h, const uint8_t *const mask) {
    if (w % 8 == 0)
        hwy_blend_impl(dst, dst_stride, tmp, w, h, mask,
                       std::integral_constant<int, 8>{});
    else
        hwy_blend_impl(dst, dst_stride, tmp, w, h, mask,
                       std::integral_constant<int, 0>{});
}

template <typename Pixel>
static HWY_INLINE void hwy_blend_v(Pixel *const dst,
                                   const ptrdiff_t dst_stride,
                                   const Pixel *const tmp, const int w,
                                   const int h) {
    if (((w * 3) >> 2) % 8 == 0)
        hwy_blend_v_impl(dst, dst_stride, tmp, w, h,
                         std::integral_constant<int, 8>{});
    else
        hwy_blend_v_impl(dst, dst_stride, tmp, w, h,
                         std::integral_constant<int, 0>{});
}

template <typename Pixel>
static HWY_INLINE void hwy_blend_h(Pixel *const dst,
                                   const ptrdiff_t dst_stride,
                                   const Pixel *const tmp, const int w,
                                   const int h) {
    if (w % 8 == 0)
        hwy_blend_h_impl(dst, dst_stride, tmp, w, h,
                         std::integral_constant<int, 8>{});
    else
        hwy_blend_h_impl(dst, dst_stride, tmp, w, h,
                         std::integral_constant<int, 0>{});
}

#define HWY_MC_COMPOUND_FNS(bpc, sfx, HIGHBD_SUFFIX, BD_MAX) \
void avg_##sfx(uint##bpc##_t *const dst, const ptrdiff_t dst_stride, \
               const int16_t *const tmp1, const int16_t *const tmp2, \
               const int w, const int h HIGHBD_SUFFIX) { \
    hwy_avg(dst, dst_stride, tmp1, tmp2, w, h, BD_MAX); \
} \
void w_avg_##sfx(uint##bpc##_t *const dst, const ptrdiff_t dst_stride, \
                 const int16_t *const tmp1, const int16_t *const tmp2, \
                 const int w, const int h, const int weight HIGHBD_SUFFIX) { \
    hwy_w_avg(dst, dst_stride, tmp1, tmp2, w, h, weight, BD_MAX); \
} \
void mask_##sfx(uint##bpc##_t *const dst, const ptrdiff_t dst_stride, \
                const int16_t *const tmp1, const int16_t *const tmp2, \
                const int w, const int h, const uint8_t *const mask \
                HIGHBD_SUFFIX) { \
    hwy_mask(dst, dst_stride, tmp1, tmp2, w, h, mask, BD_MAX); \
} \
void w_mask_444_##sfx(uint##bpc##_t *const dst, const ptrdiff_t dst_stride, \
                      const int16_t *const tmp1, const int16_t *const tmp2, \
                      const int w, const int h, uint8_t *const mask, \
                      const int sign HIGHBD_SUFFIX) { \
    hwy_w_mask<uint##bpc##_t, false, false>(dst, dst_stride, tmp1, tmp2, w, \
                                            h, mask, sign, BD_MAX); \
} \
void w_mask_422_##sfx(uint##bpc##_t *const dst, const ptrdiff_t dst_stride, \
                      const int16_t *const tmp1, const int16_t *const tmp2, \
                      const int w, const int h, uint8_t *const mask, \
                      const int sign HIGHBD_SUFFIX) { \
    hwy_w_mask<uint##bpc##_t, true, false>(dst, dst_stride, tmp1, tmp2, w, \
                                           h, mask, sign, BD_MAX); \
} \
void w_mask_420_##sfx(uint##bpc##_t *const dst, const ptrdiff_t dst_stride, \
                      const int16_t *const tmp1, const int16_t *const tmp2, \
                      const int w, const int h, uint8_t *const mask, \
                      const int sign HIGHBD_SUFFIX) { \
    hwy_w_mask<uint##bpc##_t, true, true>(dst, dst_stride, tmp1, tmp2, w, \
                                          h, mask, sign, BD_MAX); \
} \
void blend_##sfx(uint##bpc##_t *const dst, const ptrdiff_t dst_stride, \
                 const uint##bpc##_t *const tmp, const int w, const int h, \
                 const uint8_t *const mask) { \
    hwy_blend(dst, dst_stride, tmp, w, h, mask); \
} \
void blend_v_##sfx(uint##bpc##_t *const dst, const ptrdiff_t dst_stride, \
                   const uint##bpc##_t *const tmp, const int w, const int h) { \
    hwy_blend_v(dst, dst_stride, tmp, w, h); \
} \
void blend_h_##sfx(uint##bpc##_t *const dst, const ptrdiff_t dst_stride, \
                   const uint##bpc##_t *const tmp, const int w, const int h) { \
    hwy_blend_h(dst, dst_stride, tmp, w, h); \
} \
void emu_edge_##sfx(const intptr_t bw, const intptr_t bh, const intptr_t iw, \
                    const intptr_t ih, const intptr_t x, const intptr_t y, \
                    uint##bpc##_t *const dst, const ptrdiff_t dst_stride, \
                    const uint##bpc##_t *const ref, \
                    const ptrdiff_t ref_stride) { \
    hwy_emu_edge(bw, bh, iw, ih, x, y, dst, dst_stride, ref, ref_stride); \
} \
void resize_##sfx(uint##bpc##_t *const dst, const ptrdiff_t dst_stride, \
                  const uint##bpc##_t *const src, const ptrdiff_t src_stride, \
                  const int dst_w, const int h, const int src_w, const int dx, \
                  const int mx0 HIGHBD_SUFFIX) { \
    hwy_resize(dst, dst_stride, src, src_stride, dst_w, h, src_w, dx, mx0, \
               BD_MAX); \
}

#define HIGHBD_SUFFIX
#define BD_MAX 255
HWY_MC_COMPOUND_FNS(8, 8bpc, HIGHBD_SUFFIX, BD_MAX)
#undef HIGHBD_SUFFIX
#undef BD_MAX
#define HIGHBD_SUFFIX , const int bitdepth_max
#define BD_MAX bitdepth_max
HWY_MC_COMPOUND_FNS(16, 16bpc, HIGHBD_SUFFIX, BD_MAX)
#undef HIGHBD_SUFFIX
#undef BD_MAX
#undef HWY_MC_COMPOUND_FNS

}  // namespace HWY_NAMESPACE
}  // namespace dav1d

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace dav1d {
HWY_EXPORT(avg_8bpc);
HWY_EXPORT(w_avg_8bpc);
HWY_EXPORT(mask_8bpc);
HWY_EXPORT(w_mask_444_8bpc);
HWY_EXPORT(w_mask_422_8bpc);
HWY_EXPORT(w_mask_420_8bpc);
HWY_EXPORT(blend_8bpc);
HWY_EXPORT(blend_v_8bpc);
HWY_EXPORT(blend_h_8bpc);
HWY_EXPORT(emu_edge_8bpc);
HWY_EXPORT(resize_8bpc);
HWY_EXPORT(avg_16bpc);
HWY_EXPORT(w_avg_16bpc);
HWY_EXPORT(mask_16bpc);
HWY_EXPORT(w_mask_444_16bpc);
HWY_EXPORT(w_mask_422_16bpc);
HWY_EXPORT(w_mask_420_16bpc);
HWY_EXPORT(blend_16bpc);
HWY_EXPORT(blend_v_16bpc);
HWY_EXPORT(blend_h_16bpc);
HWY_EXPORT(emu_edge_16bpc);
HWY_EXPORT(resize_16bpc);
}  // namespace dav1d

namespace {
// Full Dav1dMCDSPContext layout (src/mc.h), so that this file does not need
// dav1d's bitdepth-templated C headers; only avg..resize (minus the warp
// entries) are installed here.
using McFn8 = void (*)(uint8_t *, ptrdiff_t, const uint8_t *, ptrdiff_t,
                       int, int, int, int);
using McScaledFn8 = void (*)(uint8_t *, ptrdiff_t, const uint8_t *, ptrdiff_t,
                             int, int, int, int, int, int);
using MctFn8 = void (*)(int16_t *, const uint8_t *, ptrdiff_t,
                        int, int, int, int);
using MctScaledFn8 = void (*)(int16_t *, const uint8_t *, ptrdiff_t,
                              int, int, int, int, int, int);
using AvgFn8 = void (*)(uint8_t *, ptrdiff_t, const int16_t *,
                        const int16_t *, int, int);
using WAvgFn8 = void (*)(uint8_t *, ptrdiff_t, const int16_t *,
                         const int16_t *, int, int, int);
using MaskFn8 = void (*)(uint8_t *, ptrdiff_t, const int16_t *,
                         const int16_t *, int, int, const uint8_t *);
using WMaskFn8 = void (*)(uint8_t *, ptrdiff_t, const int16_t *,
                          const int16_t *, int, int, uint8_t *, int);
using BlendFn8 = void (*)(uint8_t *, ptrdiff_t, const uint8_t *, int, int,
                          const uint8_t *);
using BlendDirFn8 = void (*)(uint8_t *, ptrdiff_t, const uint8_t *, int, int);
using Warp8x8Fn8 = void (*)(uint8_t *, ptrdiff_t, const uint8_t *, ptrdiff_t,
                            const int16_t *, int, int);
using Warp8x8tFn8 = void (*)(int16_t *, ptrdiff_t, const uint8_t *, ptrdiff_t,
                             const int16_t *, int, int);
using EmuEdgeFn8 = void (*)(intptr_t, intptr_t, intptr_t, intptr_t, intptr_t,
                            intptr_t, uint8_t *, ptrdiff_t, const uint8_t *,
                            ptrdiff_t);
using ResizeFn8 = void (*)(uint8_t *, ptrdiff_t, const uint8_t *, ptrdiff_t,
                           int, int, int, int, int);
struct McDSP8 {
    McFn8 mc[10];
    McScaledFn8 mc_scaled[10];
    MctFn8 mct[10];
    MctScaledFn8 mct_scaled[10];
    AvgFn8 avg;
    WAvgFn8 w_avg;
    MaskFn8 mask;
    WMaskFn8 w_mask[3];
    BlendFn8 blend;
    BlendDirFn8 blend_v;
    BlendDirFn8 blend_h;
    Warp8x8Fn8 warp8x8;
    Warp8x8tFn8 warp8x8t;
    EmuEdgeFn8 emu_edge;
    ResizeFn8 resize;
};

using McFn16 = void (*)(uint16_t *, ptrdiff_t, const uint16_t *, ptrdiff_t,
                        int, int, int, int, int);
using McScaledFn16 = void (*)(uint16_t *, ptrdiff_t, const uint16_t *,
                              ptrdiff_t, int, int, int, int, int, int, int);
using MctFn16 = void (*)(int16_t *, const uint16_t *, ptrdiff_t,
                         int, int, int, int, int);
using MctScaledFn16 = void (*)(int16_t *, const uint16_t *, ptrdiff_t,
                               int, int, int, int, int, int, int);
using AvgFn16 = void (*)(uint16_t *, ptrdiff_t, const int16_t *,
                         const int16_t *, int, int, int);
using WAvgFn16 = void (*)(uint16_t *, ptrdiff_t, const int16_t *,
                          const int16_t *, int, int, int, int);
using MaskFn16 = void (*)(uint16_t *, ptrdiff_t, const int16_t *,
                          const int16_t *, int, int, const uint8_t *, int);
using WMaskFn16 = void (*)(uint16_t *, ptrdiff_t, const int16_t *,
                           const int16_t *, int, int, uint8_t *, int, int);
using BlendFn16 = void (*)(uint16_t *, ptrdiff_t, const uint16_t *, int, int,
                           const uint8_t *);
using BlendDirFn16 = void (*)(uint16_t *, ptrdiff_t, const uint16_t *,
                              int, int);
using Warp8x8Fn16 = void (*)(uint16_t *, ptrdiff_t, const uint16_t *,
                             ptrdiff_t, const int16_t *, int, int, int);
using Warp8x8tFn16 = void (*)(int16_t *, ptrdiff_t, const uint16_t *,
                              ptrdiff_t, const int16_t *, int, int, int);
using EmuEdgeFn16 = void (*)(intptr_t, intptr_t, intptr_t, intptr_t, intptr_t,
                             intptr_t, uint16_t *, ptrdiff_t,
                             const uint16_t *, ptrdiff_t);
using ResizeFn16 = void (*)(uint16_t *, ptrdiff_t, const uint16_t *,
                            ptrdiff_t, int, int, int, int, int, int);
struct McDSP16 {
    McFn16 mc[10];
    McScaledFn16 mc_scaled[10];
    MctFn16 mct[10];
    MctScaledFn16 mct_scaled[10];
    AvgFn16 avg;
    WAvgFn16 w_avg;
    MaskFn16 mask;
    WMaskFn16 w_mask[3];
    BlendFn16 blend;
    BlendDirFn16 blend_v;
    BlendDirFn16 blend_h;
    Warp8x8Fn16 warp8x8;
    Warp8x8tFn16 warp8x8t;
    EmuEdgeFn16 emu_edge;
    ResizeFn16 resize;
};
}  // namespace

namespace dav1d {

// w_mask[] order in Dav1dMCDSPContext: 444, 422, 420.
static void mc_compound_dsp_init_8bpc_hwy(void *const c) {
    auto *const ctx = static_cast<McDSP8 *>(c);
    ctx->avg = HWY_DYNAMIC_POINTER(avg_8bpc);
    ctx->w_avg = HWY_DYNAMIC_POINTER(w_avg_8bpc);
    ctx->mask = HWY_DYNAMIC_POINTER(mask_8bpc);
    ctx->w_mask[0] = HWY_DYNAMIC_POINTER(w_mask_444_8bpc);
    ctx->w_mask[1] = HWY_DYNAMIC_POINTER(w_mask_422_8bpc);
    ctx->w_mask[2] = HWY_DYNAMIC_POINTER(w_mask_420_8bpc);
    ctx->blend = HWY_DYNAMIC_POINTER(blend_8bpc);
    ctx->blend_v = HWY_DYNAMIC_POINTER(blend_v_8bpc);
    ctx->blend_h = HWY_DYNAMIC_POINTER(blend_h_8bpc);
    ctx->emu_edge = HWY_DYNAMIC_POINTER(emu_edge_8bpc);
    ctx->resize = HWY_DYNAMIC_POINTER(resize_8bpc);
}

static void mc_compound_dsp_init_16bpc_hwy(void *const c) {
    auto *const ctx = static_cast<McDSP16 *>(c);
    ctx->avg = HWY_DYNAMIC_POINTER(avg_16bpc);
    ctx->w_avg = HWY_DYNAMIC_POINTER(w_avg_16bpc);
    ctx->mask = HWY_DYNAMIC_POINTER(mask_16bpc);
    ctx->w_mask[0] = HWY_DYNAMIC_POINTER(w_mask_444_16bpc);
    ctx->w_mask[1] = HWY_DYNAMIC_POINTER(w_mask_422_16bpc);
    ctx->w_mask[2] = HWY_DYNAMIC_POINTER(w_mask_420_16bpc);
    ctx->blend = HWY_DYNAMIC_POINTER(blend_16bpc);
    ctx->blend_v = HWY_DYNAMIC_POINTER(blend_v_16bpc);
    ctx->blend_h = HWY_DYNAMIC_POINTER(blend_h_16bpc);
    ctx->emu_edge = HWY_DYNAMIC_POINTER(emu_edge_16bpc);
    ctx->resize = HWY_DYNAMIC_POINTER(resize_16bpc);
}

}  // namespace dav1d

extern "C" void dav1d_mc_compound_dsp_init_hwy_8bpc(void *const c) {
    dav1d::hwy_init_chosen_target();
    dav1d::mc_compound_dsp_init_8bpc_hwy(c);
}

extern "C" void dav1d_mc_compound_dsp_init_hwy_16bpc(void *const c) {
    dav1d::hwy_init_chosen_target();
    dav1d::mc_compound_dsp_init_16bpc_hwy(c);
}

#endif  // HWY_ONCE
