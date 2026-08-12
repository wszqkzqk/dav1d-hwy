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
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Loop filter (src/loopfilter_tmpl.c) implemented with Google Highway: one
// source is compiled per SIMD target and the best one supported by the CPU is
// selected at runtime (HWY_DYNAMIC_DISPATCH). Bit-exact with the C code.

#include <stddef.h>
#include <stdint.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/hwy/loopfilter.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "src/hwy/common.h"

// Mirror of Av1FilterLUT (src/lf_mask.h); member ALIGN(16) does not change
// the offsets (e @ 0, i @ 64, sharp @ 128). Guarded: this file is re-included
// once per SIMD target.
#ifndef DAV1D_HWY_LOOPFILTER_LUTM
#define DAV1D_HWY_LOOPFILTER_LUTM
struct Av1FilterLUTM {
    uint8_t e[64];
    uint8_t i[64];
    uint64_t sharp[2];
};
#endif

HWY_BEFORE_NAMESPACE();

namespace dav1d {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// One loop_filter() call from src/loopfilter_tmpl.c: filters 4 pixels along
// the edge normal (the "taps"), for 4 independent edge positions (the lanes).
// kLanesContig selects the lane layout: true for the row-edge (v) filter
// where lanes are contiguous in memory (loads/stores are plain 4-pixel
// LoadN/StoreN), false for the col-edge (h) filter where lanes are strided
// by the row stride (handled via an in-register 4x16 transpose where
// possible). All arithmetic is per-lane in int16 lanes; only lanes [0, 4)
// are meaningful.
//
// Range proofs (pixels <= 4095): every abs() difference <= 4095; the E
// comparison sums to <= 2*4095 + 2047 = 10237; 3*(q0-p0) + clip <= 14332;
// flat8/flat6 sums <= 8*4095 + 4 = 32764 (int16-safe, arithmetic >>3);
// flat16 sums <= 16*4095 + 8 = 65528 (uint16-safe, so the >>4 must be a
// logical shift: wrapping int16 adds are bit-identical to uint16 adds).
template <typename Pixel, bool kLanesContig>
static void hwy_loop_filter4(Pixel *const dst, const ptrdiff_t lane_stride,
                             const ptrdiff_t tap_stride,
                             const int E, const int I, const int H,
                             const int wd, const int bitdepth_min_8)
{
    const hn::ScalableTag<int16_t> d;
    const hn::Rebind<uint16_t, decltype(d)> du;
    const hn::Rebind<Pixel, decltype(d)> dp;
    using V = hn::VFromD<decltype(d)>;
    constexpr size_t kN = 4;

    // Widens loaded pixels to int16 lanes (pixel values <= 4095).
    const auto load_px = [&](const Pixel *const p) {
        if constexpr (sizeof(Pixel) == 1) {
            return hn::BitCast(d, hn::PromoteTo(du, hn::LoadN(dp, p, kN)));
        } else {
            return hn::BitCast(d, hn::LoadN(dp, p, kN));
        }
    };

    // For the col-edge (h) filter the 4 lanes are strided rows. On 128-bit
    // vector targets the 4x16 pixel block around the edge (cols -8..+7) can
    // be loaded once with full vector loads and transposed in registers into
    // per-tap vectors; modified taps are queued back and flushed through the
    // inverse transpose. Cols -8/+7 beyond the C footprint are read but never
    // written (the arch asm loads full vectors the same way).
    constexpr bool kUseTranspose = !kLanesContig && sizeof(Pixel) == 2 &&
                                   HWY_LANES(int16_t) == 8;
    [[maybe_unused]] V taps[16];  // column k at index k + 8
    [[maybe_unused]] unsigned store_blocks = 0;  // bit i: cols [-6+4*i, -3+4*i]
    if constexpr (kUseTranspose) {
        const hn::Half<decltype(d)> d4;
        const hn::Repartition<int32_t, decltype(d)> d32;
        const auto load8 = [&](const Pixel *const p) {
            if constexpr (sizeof(Pixel) == 1) {
                return hn::BitCast(d, hn::PromoteTo(du, hn::LoadN(dp, p, 8)));
            } else {
                return hn::BitCast(d, hn::LoadN(dp, p, 8));
            }
        };
        for (int half = 0; half < 2; half++) {
            const ptrdiff_t col = half * 8 - 8;
            const auto r0 = load8(dst + col);
            const auto r1 = load8(dst + lane_stride + col);
            const auto r2 = load8(dst + 2 * lane_stride + col);
            const auto r3 = load8(dst + 3 * lane_stride + col);
            const auto ab_lo = hn::InterleaveLower(d, r0, r1);
            const auto ab_hi = hn::InterleaveUpper(d, r0, r1);
            const auto cd_lo = hn::InterleaveLower(d, r2, r3);
            const auto cd_hi = hn::InterleaveUpper(d, r2, r3);
            const auto c01 = hn::BitCast(d, hn::InterleaveLower(d32,
                hn::BitCast(d32, ab_lo), hn::BitCast(d32, cd_lo)));
            const auto c23 = hn::BitCast(d, hn::InterleaveUpper(d32,
                hn::BitCast(d32, ab_lo), hn::BitCast(d32, cd_lo)));
            const auto c45 = hn::BitCast(d, hn::InterleaveLower(d32,
                hn::BitCast(d32, ab_hi), hn::BitCast(d32, cd_hi)));
            const auto c67 = hn::BitCast(d, hn::InterleaveUpper(d32,
                hn::BitCast(d32, ab_hi), hn::BitCast(d32, cd_hi)));
            V *const t = taps + half * 8;
            t[0] = hn::ZeroExtendVector(d, hn::LowerHalf(d4, c01));
            t[1] = hn::ZeroExtendVector(d, hn::UpperHalf(d4, c01));
            t[2] = hn::ZeroExtendVector(d, hn::LowerHalf(d4, c23));
            t[3] = hn::ZeroExtendVector(d, hn::UpperHalf(d4, c23));
            t[4] = hn::ZeroExtendVector(d, hn::LowerHalf(d4, c45));
            t[5] = hn::ZeroExtendVector(d, hn::UpperHalf(d4, c45));
            t[6] = hn::ZeroExtendVector(d, hn::LowerHalf(d4, c67));
            t[7] = hn::ZeroExtendVector(d, hn::UpperHalf(d4, c67));
        }
    }

    const auto load_tap = [&](const ptrdiff_t k) -> V {
        const Pixel *const p = dst + k * tap_stride;
        if constexpr (kLanesContig) {
            return load_px(p);
        } else if constexpr (kUseTranspose) {
            return taps[k + 8];
        } else {
            alignas(16) Pixel tmp[kN];
            tmp[0] = p[0];
            tmp[1] = p[lane_stride];
            tmp[2] = p[2 * lane_stride];
            tmp[3] = p[3 * lane_stride];
            return load_px(tmp);
        }
    };
    // Values stored are always in [0, bitdepth_max], so demotion (8bpc) and
    // bitcast (16bpc) are exact.
    const auto store_tap = [&](const ptrdiff_t k, const V v) {
        Pixel *const p = dst + k * tap_stride;
        if constexpr (kLanesContig) {
            if constexpr (sizeof(Pixel) == 1) {
                hn::StoreN(hn::DemoteTo(dp, v), dp, p, kN);
            } else {
                hn::StoreN(hn::BitCast(dp, v), dp, p, kN);
            }
        } else if constexpr (kUseTranspose) {
            taps[k + 8] = v;
            store_blocks |= 1u << ((k + 6) / 4);
        } else {
            alignas(16) Pixel tmp[kN];
            if constexpr (sizeof(Pixel) == 1) {
                hn::StoreN(hn::DemoteTo(dp, v), dp, tmp, kN);
            } else {
                hn::StoreN(hn::BitCast(dp, v), dp, tmp, kN);
            }
            p[0] = tmp[0];
            p[lane_stride] = tmp[1];
            p[2 * lane_stride] = tmp[2];
            p[3 * lane_stride] = tmp[3];
        }
    };

    // Inverse transpose of the queued 4-column output blocks (cols -6..-3,
    // -2..+1, +2..+5); unmodified columns in a block carry their loaded
    // values. No-op when nothing was queued (e.g. the AllFalse(fm) skip).
    const auto flush_taps = [&]() {
        if constexpr (kUseTranspose) {
            const hn::Half<decltype(d)> d4;
            const hn::Repartition<int32_t, decltype(d4)> d32;
            const hn::Rebind<Pixel, decltype(d4)> dp4;
            for (int b = 0; b < 3; b++) {
                if (!(store_blocks & (1u << b))) continue;
                const V *const t = taps + 2 + b * 4;
                const auto o0 = hn::LowerHalf(d4, t[0]);
                const auto o1 = hn::LowerHalf(d4, t[1]);
                const auto o2 = hn::LowerHalf(d4, t[2]);
                const auto o3 = hn::LowerHalf(d4, t[3]);
                const auto u0 = hn::InterleaveLower(d4, o0, o1);
                const auto u1 = hn::InterleaveUpper(d4, o0, o1);
                const auto u2 = hn::InterleaveLower(d4, o2, o3);
                const auto u3 = hn::InterleaveUpper(d4, o2, o3);
                const hn::VFromD<decltype(d4)> rows[4] = {
                    hn::BitCast(d4, hn::InterleaveLower(d32,
                        hn::BitCast(d32, u0), hn::BitCast(d32, u2))),
                    hn::BitCast(d4, hn::InterleaveUpper(d32,
                        hn::BitCast(d32, u0), hn::BitCast(d32, u2))),
                    hn::BitCast(d4, hn::InterleaveLower(d32,
                        hn::BitCast(d32, u1), hn::BitCast(d32, u3))),
                    hn::BitCast(d4, hn::InterleaveUpper(d32,
                        hn::BitCast(d32, u1), hn::BitCast(d32, u3))),
                };
                const ptrdiff_t col = b * 4 - 6;
                for (int i = 0; i < 4; i++) {
                    Pixel *const p = dst + i * lane_stride + col;
                    if constexpr (sizeof(Pixel) == 1) {
                        hn::StoreN(hn::DemoteTo(dp4, rows[i]), dp4, p, 4);
                    } else {
                        hn::StoreN(hn::BitCast(dp4, rows[i]), dp4, p, 4);
                    }
                }
            }
        }
    };

    const int bitdepth_max = (256 << bitdepth_min_8) - 1;
    const auto vE = hn::Set(d, E << bitdepth_min_8);
    const auto vI = hn::Set(d, I << bitdepth_min_8);
    const auto vH = hn::Set(d, H << bitdepth_min_8);
    const auto vF = hn::Set(d, 1 << bitdepth_min_8);
    const auto vbmax = hn::Set(d, bitdepth_max);
    const auto vclip_lo = hn::Set(d, -128 << bitdepth_min_8);
    const auto vclip_hi = hn::Set(d, (128 << bitdepth_min_8) - 1);
    const auto absd = [](const V a, const V b) { return hn::Abs(hn::Sub(a, b)); };

    const V p1 = load_tap(-2), p0 = load_tap(-1);
    const V q0 = load_tap(0), q1 = load_tap(1);

    auto fm = hn::And(hn::And(hn::Le(absd(p1, p0), vI), hn::Le(absd(q1, q0), vI)),
                      hn::Le(hn::Add(hn::Add(absd(p0, q0), absd(p0, q0)),
                                     hn::ShiftRight<1>(absd(p1, q1))), vE));
    // Inert lanes [4, Lanes) contain zeros and would pass all the fm tests;
    // mask them off so AllFalse() can skip groups where nothing is filtered.
    fm = hn::And(fm, hn::FirstN(d, kN));
    V p2 = hn::Zero(d), q2 = hn::Zero(d);
    V p3 = hn::Zero(d), q3 = hn::Zero(d);
    if (wd > 4) {
        p2 = load_tap(-3);
        q2 = load_tap(2);
        fm = hn::And(fm, hn::And(hn::Le(absd(p2, p1), vI),
                                 hn::Le(absd(q2, q1), vI)));
    }
    if (wd > 6) {
        p3 = load_tap(-4);
        q3 = load_tap(3);
        fm = hn::And(fm, hn::And(hn::Le(absd(p3, p2), vI),
                                 hn::Le(absd(q3, q2), vI)));
    }
    if (hn::AllFalse(d, fm)) return;

    const auto iclip_diff = [&](const V v) { return hn::Clamp(v, vclip_lo, vclip_hi); };
    const auto iclip_px = [&](const V v) { return hn::Clamp(v, hn::Zero(d), vbmax); };

    // hev/normal 2- or 4-tap filter, blended in at the lanes selected by m
    // (m always includes fm, so accumulators can start at the originals and
    // the C "continue" needs no further gating at store time).
    const auto hev_filter = [&](const auto m, V &o_p1, V &o_p0, V &o_q0, V &o_q1) {
        const auto hev = hn::Or(hn::Gt(absd(p1, p0), vH), hn::Gt(absd(q1, q0), vH));
        const auto f0 = hn::Mul(hn::Sub(q0, p0), hn::Set(d, 3));
        const auto f = hn::IfThenElse(hev,
            iclip_diff(hn::Add(f0, iclip_diff(hn::Sub(p1, q1)))), iclip_diff(f0));
        const auto f1 = hn::ShiftRight<3>(hn::Min(hn::Add(f, hn::Set(d, 4)), vclip_hi));
        const auto f2 = hn::ShiftRight<3>(hn::Min(hn::Add(f, hn::Set(d, 3)), vclip_hi));
        const auto f3 = hn::ShiftRight<1>(hn::Add(f1, hn::Set(d, 1)));
        o_p1 = hn::IfThenElse(m, hn::IfThenElse(hev, p1, iclip_px(hn::Add(p1, f3))), o_p1);
        o_p0 = hn::IfThenElse(m, iclip_px(hn::Add(p0, f2)), o_p0);
        o_q0 = hn::IfThenElse(m, iclip_px(hn::Sub(q0, f1)), o_q0);
        o_q1 = hn::IfThenElse(m, hn::IfThenElse(hev, q1, iclip_px(hn::Sub(q1, f3))), o_q1);
    };

    if (wd >= 8) {
        const auto flat8in = hn::And(
            hn::And(hn::And(hn::Le(absd(p2, p0), vF), hn::Le(absd(p1, p0), vF)),
                    hn::And(hn::Le(absd(q1, q0), vF), hn::Le(absd(q2, q0), vF))),
            hn::And(hn::Le(absd(p3, p0), vF), hn::Le(absd(q3, q0), vF)));
        const auto mfin = hn::And(fm, flat8in);
        auto m8 = mfin;
        V o_p2 = p2, o_p1 = p1, o_p0 = p0, o_q0 = q0, o_q1 = q1, o_q2 = q2;
        if (wd >= 16) {
            const V p4 = load_tap(-5), p5 = load_tap(-6), p6 = load_tap(-7);
            const V q4 = load_tap(4), q5 = load_tap(5), q6 = load_tap(6);
            const auto flat8out = hn::And(
                hn::And(hn::And(hn::Le(absd(p6, p0), vF), hn::Le(absd(p5, p0), vF)),
                        hn::Le(absd(p4, p0), vF)),
                hn::And(hn::And(hn::Le(absd(q4, q0), vF), hn::Le(absd(q5, q0), vF)),
                        hn::Le(absd(q6, q0), vF)));
            const auto m16 = hn::And(mfin, flat8out);
            m8 = hn::AndNot(m16, mfin);
            // The 12-tap path is rare; skip its sums when no lane selects it.
            if (!hn::AllFalse(d, m16)) {
                const auto v2c = hn::Set(d, 2);
                const auto v16r = hn::Set(d, 8);
                // Logical >>4 on the uint16 bit pattern (see range proof above).
                const auto shr4 = [&](const V sum) {
                    return hn::BitCast(d, hn::ShiftRight<4>(hn::BitCast(du, sum)));
                };
                const auto s16_p5 = shr4(hn::Add(hn::Add(hn::Add(
                    hn::Mul(p6, hn::Set(d, 7)), hn::Add(hn::Mul(p5, v2c), hn::Mul(p4, v2c))),
                    hn::Add(hn::Add(p3, p2), hn::Add(p1, p0))), hn::Add(q0, v16r)));
                const auto s16_p4 = shr4(hn::Add(hn::Add(hn::Add(
                    hn::Mul(p6, hn::Set(d, 5)), hn::Add(hn::Mul(p5, v2c), hn::Mul(p4, v2c))),
                    hn::Add(hn::Add(hn::Mul(p3, v2c), p2), hn::Add(p1, p0))),
                    hn::Add(hn::Add(q0, q1), v16r)));
                const auto s16_p3 = shr4(hn::Add(hn::Add(hn::Add(
                    hn::Mul(p6, hn::Set(d, 4)), hn::Add(p5, hn::Mul(p4, v2c))),
                    hn::Add(hn::Add(hn::Mul(p3, v2c), hn::Mul(p2, v2c)), hn::Add(p1, p0))),
                    hn::Add(hn::Add(q0, q1), hn::Add(q2, v16r))));
                const auto s16_p2 = shr4(hn::Add(hn::Add(hn::Add(
                    hn::Mul(p6, hn::Set(d, 3)), hn::Add(hn::Add(p5, p4), hn::Mul(p3, v2c))),
                    hn::Add(hn::Add(hn::Mul(p2, v2c), hn::Mul(p1, v2c)), hn::Add(p0, q0))),
                    hn::Add(hn::Add(q1, q2), hn::Add(q3, v16r))));
                const auto s16_p1 = shr4(hn::Add(hn::Add(hn::Add(
                    hn::Mul(p6, v2c), hn::Add(hn::Add(p5, p4), hn::Add(p3, hn::Mul(p2, v2c)))),
                    hn::Add(hn::Add(hn::Mul(p1, v2c), hn::Mul(p0, v2c)), hn::Add(q0, q1))),
                    hn::Add(hn::Add(q2, q3), hn::Add(q4, v16r))));
                const auto s16_p0 = shr4(hn::Add(hn::Add(hn::Add(
                    hn::Add(p6, p5), hn::Add(hn::Add(p4, p3), hn::Add(p2, hn::Mul(p1, v2c)))),
                    hn::Add(hn::Add(hn::Mul(p0, v2c), hn::Mul(q0, v2c)), hn::Add(q1, q2))),
                    hn::Add(hn::Add(q3, q4), hn::Add(q5, v16r))));
                const auto s16_q0 = shr4(hn::Add(hn::Add(hn::Add(
                    hn::Add(p5, p4), hn::Add(hn::Add(p3, p2), hn::Add(p1, hn::Mul(p0, v2c)))),
                    hn::Add(hn::Add(hn::Mul(q0, v2c), hn::Mul(q1, v2c)), hn::Add(q2, q3))),
                    hn::Add(hn::Add(q4, q5), hn::Add(q6, v16r))));
                const auto s16_q1 = shr4(hn::Add(hn::Add(hn::Add(
                    hn::Add(p4, p3), hn::Add(hn::Add(p2, p1), hn::Add(p0, hn::Mul(q0, v2c)))),
                    hn::Add(hn::Add(hn::Mul(q1, v2c), hn::Mul(q2, v2c)), hn::Add(q3, q4))),
                    hn::Add(hn::Add(q5, hn::Mul(q6, v2c)), v16r)));
                const auto s16_q2 = shr4(hn::Add(hn::Add(hn::Add(
                    hn::Add(p3, p2), hn::Add(hn::Add(p1, p0), hn::Add(q0, hn::Mul(q1, v2c)))),
                    hn::Add(hn::Add(hn::Mul(q2, v2c), hn::Mul(q3, v2c)), hn::Add(q4, q5))),
                    hn::Add(hn::Mul(q6, hn::Set(d, 3)), v16r)));
                const auto s16_q3 = shr4(hn::Add(hn::Add(hn::Add(
                    hn::Add(p2, p1), hn::Add(hn::Add(p0, q0), hn::Add(q1, hn::Mul(q2, v2c)))),
                    hn::Add(hn::Add(hn::Mul(q3, v2c), hn::Mul(q4, v2c)), q5)),
                    hn::Add(hn::Mul(q6, hn::Set(d, 4)), v16r)));
                const auto s16_q4 = shr4(hn::Add(hn::Add(hn::Add(
                    hn::Add(p1, p0), hn::Add(hn::Add(q0, q1), hn::Add(q2, hn::Mul(q3, v2c)))),
                    hn::Add(hn::Add(hn::Mul(q4, v2c), hn::Mul(q5, v2c)), hn::Mul(q6, hn::Set(d, 5)))),
                    v16r));
                const auto s16_q5 = shr4(hn::Add(hn::Add(hn::Add(
                    hn::Add(p0, q0), hn::Add(hn::Add(q1, q2), hn::Add(q3, hn::Mul(q4, v2c)))),
                    hn::Add(hn::Mul(q5, v2c), hn::Mul(q6, hn::Set(d, 7)))),
                    v16r));
                store_tap(-6, hn::IfThenElse(m16, s16_p5, p5));
                store_tap(-5, hn::IfThenElse(m16, s16_p4, p4));
                store_tap(-4, hn::IfThenElse(m16, s16_p3, p3));
                o_p2 = hn::IfThenElse(m16, s16_p2, o_p2);
                o_p1 = hn::IfThenElse(m16, s16_p1, o_p1);
                o_p0 = hn::IfThenElse(m16, s16_p0, o_p0);
                o_q0 = hn::IfThenElse(m16, s16_q0, o_q0);
                o_q1 = hn::IfThenElse(m16, s16_q1, o_q1);
                o_q2 = hn::IfThenElse(m16, s16_q2, o_q2);
                store_tap(+3, hn::IfThenElse(m16, s16_q3, q3));
                store_tap(+4, hn::IfThenElse(m16, s16_q4, q4));
                store_tap(+5, hn::IfThenElse(m16, s16_q5, q5));
            }
        }
        // 8-tap path for the flat lanes not taking the 12-tap path.
        if (!hn::AllFalse(d, m8)) {
            const auto v4r = hn::Set(d, 4);
            const auto s8_p2 = hn::ShiftRight<3>(hn::Add(hn::Add(hn::Add(
                hn::Mul(p3, hn::Set(d, 3)), hn::Mul(p2, hn::Set(d, 2))),
                hn::Add(p1, p0)), hn::Add(q0, v4r)));
            const auto s8_p1 = hn::ShiftRight<3>(hn::Add(hn::Add(hn::Add(
                hn::Mul(p3, hn::Set(d, 2)), hn::Add(p2, hn::Mul(p1, hn::Set(d, 2)))),
                hn::Add(p0, q0)), hn::Add(q1, v4r)));
            const auto s8_p0 = hn::ShiftRight<3>(hn::Add(hn::Add(hn::Add(
                hn::Add(p3, p2), hn::Add(p1, hn::Mul(p0, hn::Set(d, 2)))),
                hn::Add(q0, q1)), hn::Add(q2, v4r)));
            const auto s8_q0 = hn::ShiftRight<3>(hn::Add(hn::Add(hn::Add(
                hn::Add(p2, p1), hn::Add(p0, hn::Mul(q0, hn::Set(d, 2)))),
                hn::Add(q1, q2)), hn::Add(q3, v4r)));
            const auto s8_q1 = hn::ShiftRight<3>(hn::Add(hn::Add(hn::Add(
                hn::Add(p1, p0), hn::Add(q0, hn::Mul(q1, hn::Set(d, 2)))),
                hn::Add(q2, hn::Mul(q3, hn::Set(d, 2)))), v4r));
            const auto s8_q2 = hn::ShiftRight<3>(hn::Add(hn::Add(hn::Add(
                hn::Add(p0, q0), hn::Add(q1, hn::Mul(q2, hn::Set(d, 2)))),
                hn::Mul(q3, hn::Set(d, 3))), v4r));
            o_p2 = hn::IfThenElse(m8, s8_p2, o_p2);
            o_p1 = hn::IfThenElse(m8, s8_p1, o_p1);
            o_p0 = hn::IfThenElse(m8, s8_p0, o_p0);
            o_q0 = hn::IfThenElse(m8, s8_q0, o_q0);
            o_q1 = hn::IfThenElse(m8, s8_q1, o_q1);
            o_q2 = hn::IfThenElse(m8, s8_q2, o_q2);
        }
        const auto mh = hn::AndNot(mfin, fm);
        if (!hn::AllFalse(d, mh)) hev_filter(mh, o_p1, o_p0, o_q0, o_q1);
        store_tap(-3, o_p2);
        store_tap(-2, o_p1);
        store_tap(-1, o_p0);
        store_tap(+0, o_q0);
        store_tap(+1, o_q1);
        store_tap(+2, o_q2);
        flush_taps();
        return;
    }

    if (wd == 6) {
        const auto flat8in = hn::And(hn::And(hn::Le(absd(p2, p0), vF),
                                             hn::Le(absd(p1, p0), vF)),
                                     hn::And(hn::Le(absd(q1, q0), vF),
                                             hn::Le(absd(q2, q0), vF)));
        const auto mfin = hn::And(fm, flat8in);
        V o_p1 = p1, o_p0 = p0, o_q0 = q0, o_q1 = q1;
        if (!hn::AllFalse(d, mfin)) {
            const auto v8r = hn::Set(d, 4);
            const auto s6_p1 = hn::ShiftRight<3>(hn::Add(hn::Add(hn::Add(
                hn::Add(hn::Mul(p2, hn::Set(d, 3)), hn::Mul(p1, hn::Set(d, 2))),
                hn::Mul(p0, hn::Set(d, 2))), q0), v8r));
            const auto s6_p0 = hn::ShiftRight<3>(hn::Add(hn::Add(hn::Add(hn::Add(p2,
                hn::Mul(p1, hn::Set(d, 2))), hn::Mul(p0, hn::Set(d, 2))),
                hn::Mul(q0, hn::Set(d, 2))), hn::Add(q1, v8r)));
            const auto s6_q0 = hn::ShiftRight<3>(hn::Add(hn::Add(hn::Add(hn::Add(p1,
                hn::Mul(p0, hn::Set(d, 2))), hn::Mul(q0, hn::Set(d, 2))),
                hn::Mul(q1, hn::Set(d, 2))), hn::Add(q2, v8r)));
            const auto s6_q1 = hn::ShiftRight<3>(hn::Add(hn::Add(hn::Add(hn::Add(p0,
                hn::Mul(q0, hn::Set(d, 2))), hn::Mul(q1, hn::Set(d, 2))),
                hn::Mul(q2, hn::Set(d, 2))), hn::Add(q2, v8r)));
            o_p1 = hn::IfThenElse(mfin, s6_p1, o_p1);
            o_p0 = hn::IfThenElse(mfin, s6_p0, o_p0);
            o_q0 = hn::IfThenElse(mfin, s6_q0, o_q0);
            o_q1 = hn::IfThenElse(mfin, s6_q1, o_q1);
        }
        const auto mh = hn::AndNot(mfin, fm);
        if (!hn::AllFalse(d, mh)) hev_filter(mh, o_p1, o_p0, o_q0, o_q1);
        store_tap(-2, o_p1);
        store_tap(-1, o_p0);
        store_tap(+0, o_q0);
        store_tap(+1, o_q1);
        flush_taps();
        return;
    }

    // wd == 4
    V o_p1 = p1, o_p0 = p0, o_q0 = q0, o_q1 = q1;
    hev_filter(fm, o_p1, o_p0, o_q0, o_q1);
    store_tap(-2, o_p1);
    store_tap(-1, o_p0);
    store_tap(+0, o_q0);
    store_tap(+1, o_q1);
    flush_taps();
}

// The wrappers below match their *_c counterparts in src/loopfilter_tmpl.c
// (mask iteration, level lookup and filter-width selection).

template <typename Pixel>
static void hwy_loop_filter_h_sb128y(Pixel *dst, const ptrdiff_t stride,
                                     const uint32_t *const vmask,
                                     const uint8_t (*l)[4], const ptrdiff_t b4_stride,
                                     const Av1FilterLUTM *const lut,
                                     const int bitdepth_min_8)
{
    const unsigned vm = vmask[0] | vmask[1] | vmask[2];
    const ptrdiff_t pxs = stride / (ptrdiff_t) sizeof(Pixel);
    for (unsigned y = 1; vm & ~(y - 1); y <<= 1, dst += 4 * pxs, l += b4_stride) {
        if (vm & y) {
            const int L = l[0][0] ? l[0][0] : l[-1][0];
            if (!L) continue;
            const int H = L >> 4;
            const int E = lut->e[L], I = lut->i[L];
            const int idx = (vmask[2] & y) ? 2 : !!(vmask[1] & y);
            hwy_loop_filter4<Pixel, false>(dst, pxs, 1, E, I, H, 4 << idx,
                                           bitdepth_min_8);
        }
    }
}

template <typename Pixel>
static void hwy_loop_filter_v_sb128y(Pixel *dst, const ptrdiff_t stride,
                                     const uint32_t *const vmask,
                                     const uint8_t (*l)[4], const ptrdiff_t b4_stride,
                                     const Av1FilterLUTM *const lut,
                                     const int bitdepth_min_8)
{
    const unsigned vm = vmask[0] | vmask[1] | vmask[2];
    const ptrdiff_t pxs = stride / (ptrdiff_t) sizeof(Pixel);
    for (unsigned x = 1; vm & ~(x - 1); x <<= 1, dst += 4, l++) {
        if (vm & x) {
            const int L = l[0][0] ? l[0][0] : l[-b4_stride][0];
            if (!L) continue;
            const int H = L >> 4;
            const int E = lut->e[L], I = lut->i[L];
            const int idx = (vmask[2] & x) ? 2 : !!(vmask[1] & x);
            hwy_loop_filter4<Pixel, true>(dst, 1, pxs, E, I, H, 4 << idx,
                                          bitdepth_min_8);
        }
    }
}

template <typename Pixel>
static void hwy_loop_filter_h_sb128uv(Pixel *dst, const ptrdiff_t stride,
                                      const uint32_t *const vmask,
                                      const uint8_t (*l)[4], const ptrdiff_t b4_stride,
                                      const Av1FilterLUTM *const lut,
                                      const int bitdepth_min_8)
{
    const unsigned vm = vmask[0] | vmask[1];
    const ptrdiff_t pxs = stride / (ptrdiff_t) sizeof(Pixel);
    for (unsigned y = 1; vm & ~(y - 1); y <<= 1, dst += 4 * pxs, l += b4_stride) {
        if (vm & y) {
            const int L = l[0][0] ? l[0][0] : l[-1][0];
            if (!L) continue;
            const int H = L >> 4;
            const int E = lut->e[L], I = lut->i[L];
            const int idx = !!(vmask[1] & y);
            hwy_loop_filter4<Pixel, false>(dst, pxs, 1, E, I, H, 4 + 2 * idx,
                                           bitdepth_min_8);
        }
    }
}

template <typename Pixel>
static void hwy_loop_filter_v_sb128uv(Pixel *dst, const ptrdiff_t stride,
                                      const uint32_t *const vmask,
                                      const uint8_t (*l)[4], const ptrdiff_t b4_stride,
                                      const Av1FilterLUTM *const lut,
                                      const int bitdepth_min_8)
{
    const unsigned vm = vmask[0] | vmask[1];
    const ptrdiff_t pxs = stride / (ptrdiff_t) sizeof(Pixel);
    for (unsigned x = 1; vm & ~(x - 1); x <<= 1, dst += 4, l++) {
        if (vm & x) {
            const int L = l[0][0] ? l[0][0] : l[-b4_stride][0];
            if (!L) continue;
            const int H = L >> 4;
            const int E = lut->e[L], I = lut->i[L];
            const int idx = !!(vmask[1] & x);
            hwy_loop_filter4<Pixel, true>(dst, 1, pxs, E, I, H, 4 + 2 * idx,
                                          bitdepth_min_8);
        }
    }
}

#define LOOP_FILTER_SB_FNS(bpc, sfx) \
void loop_filter_h_sb128y_##sfx(uint##bpc##_t *dst, const ptrdiff_t stride, \
        const uint32_t *const vmask, const uint8_t (*l)[4], \
        const ptrdiff_t b4_stride, const Av1FilterLUTM *const lut, \
        const int /* h */ HIGHBD_SUFFIX(bpc)) \
{ \
    hwy_loop_filter_h_sb128y<uint##bpc##_t>(dst, stride, vmask, l, b4_stride, \
                                            lut, BD_MINUS_8(bpc)); \
} \
void loop_filter_v_sb128y_##sfx(uint##bpc##_t *dst, const ptrdiff_t stride, \
        const uint32_t *const vmask, const uint8_t (*l)[4], \
        const ptrdiff_t b4_stride, const Av1FilterLUTM *const lut, \
        const int /* w */ HIGHBD_SUFFIX(bpc)) \
{ \
    hwy_loop_filter_v_sb128y<uint##bpc##_t>(dst, stride, vmask, l, b4_stride, \
                                            lut, BD_MINUS_8(bpc)); \
} \
void loop_filter_h_sb128uv_##sfx(uint##bpc##_t *dst, const ptrdiff_t stride, \
        const uint32_t *const vmask, const uint8_t (*l)[4], \
        const ptrdiff_t b4_stride, const Av1FilterLUTM *const lut, \
        const int /* h */ HIGHBD_SUFFIX(bpc)) \
{ \
    hwy_loop_filter_h_sb128uv<uint##bpc##_t>(dst, stride, vmask, l, b4_stride, \
                                             lut, BD_MINUS_8(bpc)); \
} \
void loop_filter_v_sb128uv_##sfx(uint##bpc##_t *dst, const ptrdiff_t stride, \
        const uint32_t *const vmask, const uint8_t (*l)[4], \
        const ptrdiff_t b4_stride, const Av1FilterLUTM *const lut, \
        const int /* w */ HIGHBD_SUFFIX(bpc)) \
{ \
    hwy_loop_filter_v_sb128uv<uint##bpc##_t>(dst, stride, vmask, l, b4_stride, \
                                             lut, BD_MINUS_8(bpc)); \
}

#define HIGHBD_SUFFIX(bpc)
#define BD_MINUS_8(bpc) 0
LOOP_FILTER_SB_FNS(8, 8bpc)
#undef HIGHBD_SUFFIX
#undef BD_MINUS_8
#define HIGHBD_SUFFIX(bpc) , const int bitdepth_max
#define BD_MINUS_8(bpc) hwy_ulog2(bitdepth_max) - 7
LOOP_FILTER_SB_FNS(16, 16bpc)
#undef HIGHBD_SUFFIX
#undef BD_MINUS_8
#undef LOOP_FILTER_SB_FNS

}  // namespace HWY_NAMESPACE
}  // namespace dav1d

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace dav1d {
HWY_EXPORT(loop_filter_h_sb128y_8bpc);
HWY_EXPORT(loop_filter_v_sb128y_8bpc);
HWY_EXPORT(loop_filter_h_sb128uv_8bpc);
HWY_EXPORT(loop_filter_v_sb128uv_8bpc);
HWY_EXPORT(loop_filter_h_sb128y_16bpc);
HWY_EXPORT(loop_filter_v_sb128y_16bpc);
HWY_EXPORT(loop_filter_h_sb128uv_16bpc);
HWY_EXPORT(loop_filter_v_sb128uv_16bpc);
}  // namespace dav1d

namespace {
// Mirrors of Dav1dLoopFilterDSPContext (src/loopfilter.h), so that this file
// does not need dav1d's bitdepth-templated C headers.
using LfSbFn8 = void (*)(uint8_t *, ptrdiff_t, const uint32_t *,
                         const uint8_t (*)[4], ptrdiff_t,
                         const Av1FilterLUTM *, int);
using LfSbFn16 = void (*)(uint16_t *, ptrdiff_t, const uint32_t *,
                          const uint8_t (*)[4], ptrdiff_t,
                          const Av1FilterLUTM *, int, int);
struct LfDSP8 {
    LfSbFn8 loop_filter_sb[2][2];
};
struct LfDSP16 {
    LfSbFn16 loop_filter_sb[2][2];
};
}  // namespace

namespace dav1d {

static void loop_filter_dsp_init_8bpc_hwy(void *const c) {
    auto *const ctx = static_cast<LfDSP8 *>(c);
    ctx->loop_filter_sb[0][0] = HWY_DYNAMIC_POINTER(loop_filter_h_sb128y_8bpc);
    ctx->loop_filter_sb[0][1] = HWY_DYNAMIC_POINTER(loop_filter_v_sb128y_8bpc);
    ctx->loop_filter_sb[1][0] = HWY_DYNAMIC_POINTER(loop_filter_h_sb128uv_8bpc);
    ctx->loop_filter_sb[1][1] = HWY_DYNAMIC_POINTER(loop_filter_v_sb128uv_8bpc);
}

static void loop_filter_dsp_init_16bpc_hwy(void *const c) {
    auto *const ctx = static_cast<LfDSP16 *>(c);
    ctx->loop_filter_sb[0][0] = HWY_DYNAMIC_POINTER(loop_filter_h_sb128y_16bpc);
    ctx->loop_filter_sb[0][1] = HWY_DYNAMIC_POINTER(loop_filter_v_sb128y_16bpc);
    ctx->loop_filter_sb[1][0] = HWY_DYNAMIC_POINTER(loop_filter_h_sb128uv_16bpc);
    ctx->loop_filter_sb[1][1] = HWY_DYNAMIC_POINTER(loop_filter_v_sb128uv_16bpc);
}

}  // namespace dav1d

extern "C" void dav1d_loop_filter_dsp_init_hwy_8bpc(void *const c) {
    dav1d::hwy_init_chosen_target();
    dav1d::loop_filter_dsp_init_8bpc_hwy(c);
}

extern "C" void dav1d_loop_filter_dsp_init_hwy_16bpc(void *const c) {
    dav1d::hwy_init_chosen_target();
    dav1d::loop_filter_dsp_init_16bpc_hwy(c);
}

#endif  // HWY_ONCE
