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

// Motion compensation put/prep kernels (src/mc_tmpl.c) implemented with
// Google Highway: one source is compiled per SIMD target and the best one
// supported by the CPU is selected at runtime (HWY_DYNAMIC_DISPATCH).
// Covers the plain and scaled (reference-scaling) variants and the warp8x8
// kernels. Bit-exact with the C code.
//
// Vectors are capped at 128 bits so that MulEven/MulOdd (i16 -> i32 widening
// multiplies) and the InterleaveLower/Upper + Combine recombination are
// whole-vector operations on every target (wider x86 targets interleave per
// 128-bit block, which would scramble the lane order). All filter
// intermediates fit int16/int32 by design of the AV1 subpel filters, see
// CombineEvenOdd.

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <type_traits>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/hwy/mc.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "src/hwy/common.h"

// Defined in src/tables.c.
extern "C" const int8_t dav1d_mc_subpel_filters[6][15][8];
extern "C" const int8_t dav1d_mc_warp_filter[193][8];

HWY_BEFORE_NAMESPACE();

namespace dav1d {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// enum Dav1dFilterMode in include/dav1d/headers.h.
enum {
    kFilterRegular = 0,
    kFilterSmooth  = 1,
    kFilterSharp   = 2,
};

// Pitch of the horizontal-pass intermediate, as in src/mc_tmpl.c.
constexpr int kMidStride = 128;

// i16 store for the mid buffer and prep output.
template <int kCap, class D16>
static HWY_INLINE void StoreI16(const D16 d, int16_t *const p, const size_t n,
                                const hn::VFromD<D16> v) {
    constexpr bool kFull = kCap != 0;
    if (kFull) hn::StoreU(v, d, p); else hn::StoreN(v, d, p, n);
}

// Packs i32 even/odd lane results (MulEven/MulOdd order) back into the
// original i16 column order. All filter outputs fit int16 (worst case over
// the tables in src/tables.c: |h-pass| <= 23547, |v-pass pre-clip/bias| <=
// 36985), so DemoteTo is exact.
template <class D16, class V32>
static HWY_INLINE hn::VFromD<D16> CombineEvenOdd(const D16 d, const V32 sum_e,
                                                 const V32 sum_o) {
    const hn::Repartition<int32_t, D16> d32;
    const hn::Rebind<int16_t, decltype(d32)> dh;
    const auto lo = hn::DemoteTo(dh, hn::InterleaveLower(sum_e, sum_o));
    const auto hi = hn::DemoteTo(dh, hn::InterleaveUpper(d32, sum_e, sum_o));
    return hn::Combine(d, hi, lo);
}

// (dot_e/dot_o + rnd) >> sh, packed to i16. ShiftRightSame on i32 is an
// arithmetic shift, matching the C code's >> on int.
template <class D16, class V32>
static HWY_INLINE hn::VFromD<D16> RoundPack(const D16 d, const V32 dot_e,
                                            const V32 dot_o, const int rnd,
                                            const int sh) {
    const hn::Repartition<int32_t, D16> d32;
    const auto vr = hn::Set(d32, rnd);
    return CombineEvenOdd(d, hn::ShiftRightSame(hn::Add(dot_e, vr), sh),
                          hn::ShiftRightSame(hn::Add(dot_o, vr), sh));
}

// Broadcast filter taps, built once per call.
template <class D16>
static HWY_INLINE void LoadTaps(const D16 d, const int8_t *const F,
                                hn::VFromD<D16> t[8]) {
    for (int k = 0; k < 8; k++) t[k] = hn::Set(d, F[k]);
}

// Stores rounded (dot + rnd) >> sh i32 even/odd dot results as i16 in
// column order; values fit int16 (see CombineEvenOdd), so the demotes are
// exact. Full chunks use a paired interleaving store.
template <int kCap, class D16, class V32>
static HWY_INLINE void StorePackI16(const D16 d, int16_t *const p, const size_t n,
                                    V32 dot_e, V32 dot_o, const int rnd,
                                    const int sh) {
    const hn::Repartition<int32_t, D16> d32;
    const hn::Rebind<int16_t, hn::Repartition<int32_t, D16>> dh;
    const auto vr = hn::Set(d32, rnd);
    dot_e = hn::ShiftRightSame(hn::Add(dot_e, vr), sh);
    dot_o = hn::ShiftRightSame(hn::Add(dot_o, vr), sh);
    if constexpr (kCap == 8) {
        hn::StoreInterleaved2(hn::DemoteTo(dh, dot_e), hn::DemoteTo(dh, dot_o),
                              dh, p);
    } else {
        StoreI16<kCap>(d, p, n, CombineEvenOdd(d, dot_e, dot_o));
    }
}

// Same, stored as pixels after clipping to [0, bitdepth_max]. For 8bpc the
// saturating u8 demote IS the clip (pre-clip values fit i16, see
// CombineEvenOdd); for 16bpc a Min plus the saturating u16 demote (negatives
// become 0) suffices.
template <int kCap, class D16, typename Pixel, class V32>
static HWY_INLINE void StorePackPx(const D16 d, Pixel *const p, const size_t n,
                                   V32 dot_e, V32 dot_o, const int rnd,
                                   const int sh, const int bitdepth_max) {
    const hn::Repartition<int32_t, D16> d32;
    const auto vr = hn::Set(d32, rnd);
    dot_e = hn::ShiftRightSame(hn::Add(dot_e, vr), sh);
    dot_o = hn::ShiftRightSame(hn::Add(dot_o, vr), sh);
    if constexpr (sizeof(Pixel) == 1) {
        const hn::Rebind<Pixel, D16> dp;
        const auto v = CombineEvenOdd(d, dot_e, dot_o);
        if (kCap) hn::StoreU(hn::DemoteTo(dp, v), dp, p);
        else hn::StoreN(hn::DemoteTo(dp, v), dp, p, n);
    } else if constexpr (kCap == 8) {
        const hn::Rebind<uint16_t, hn::Repartition<int32_t, D16>> dhu;
        const auto vmax = hn::Set(d32, bitdepth_max);
        hn::StoreInterleaved2(hn::DemoteTo(dhu, hn::Min(dot_e, vmax)),
                              hn::DemoteTo(dhu, hn::Min(dot_o, vmax)), dhu,
                              reinterpret_cast<uint16_t *>(p));
    } else {
        const hn::Rebind<Pixel, D16> dp;
        const auto vmax16 = hn::Set(d, bitdepth_max);
        const auto v = hn::Clamp(CombineEvenOdd(d, dot_e, dot_o),
                                 hn::Zero(d), vmax16);
        hn::StoreN(hn::BitCast(dp, v), dp, p, n);
    }
}

// Calls f(integral_constant<int, k>{}) for k in [k0, k1).
template <int k0, int k1, class F>
static HWY_INLINE void ForEachTap(const F& f) {
    if constexpr (k0 < k1) {
        f(std::integral_constant<int, k0>{});
        ForEachTap<k0 + 1, k1>(f);
    }
}

// Lane k of the window pair (v8:v0); k == 0 and k == 8 are the loads.
template <int k, class D16, class V16>
static HWY_INLINE V16 ShiftedWindow(const D16 d, const V16 v8, const V16 v0) {
    if constexpr (k == 0) {
        return v0;
    } else if constexpr (k == 8) {
        return v8;
    } else {
        return hn::CombineShiftRightBytes<2 * k>(d, v8, v0);
    }
}

// One chunk of n columns of the 8-tap horizontal filter at p (== row + x - 3):
// (sum F[k] * p[k]) + rnd >> sh; DAV1D_FILTER_8TAP_RND. k6tap skips the
// vanishing outer taps.
// Full chunks load the whole pixel window once (two loads, both within the
// range the C code reads) and derive the per-tap vectors as byte shifts; the
// odd output lanes of tap k are the even lanes of the k+1 shift, so MulOdd
// (and its per-tap lane shuffle) is never needed.
template <int kCap, bool k6tap, class D16, typename Src, class V32>
static HWY_INLINE void Dot8H(const D16 d, const Src *const p,
                             const size_t n, const hn::VFromD<D16> F[8],
                             V32& dot_e, V32& dot_o) {
    const hn::Repartition<int32_t, D16> d32;
    auto dot_e0 = hn::Zero(d32);
    auto dot_e1 = hn::Zero(d32);
    auto dot_o0 = hn::Zero(d32);
    auto dot_o1 = hn::Zero(d32);
    constexpr int k0 = k6tap ? 1 : 0;
    constexpr int k1 = k6tap ? 7 : 8;
    if constexpr (kCap == 8) {
        const auto v0 = LoadI16<kCap>(d, p, n);
        const auto v8 = LoadI16<0>(d, p + 8, k6tap ? 6 : 7);
        hn::VFromD<D16> v[k1 + 1];
        ForEachTap<0, k1 + 1>([&](auto kc) {
            v[decltype(kc)::value] = ShiftedWindow<decltype(kc)::value>(d, v8, v0);
        });
        ForEachTap<k0, k1>([&](auto kc) {
            constexpr int k = decltype(kc)::value;
            auto& de = (k & 1) ? dot_e1 : dot_e0;
            auto& dout = (k & 1) ? dot_o1 : dot_o0;
            de = hn::Add(de, hn::MulEven(v[k], F[k]));
            dout = hn::Add(dout, hn::MulEven(v[k + 1], F[k]));
        });
    } else {
        for (int k = k0; k < k1; k++) {
            const auto v = LoadI16<kCap>(d, p + k, n);
            auto& de = (k & 1) ? dot_e1 : dot_e0;
            auto& dout = (k & 1) ? dot_o1 : dot_o0;
            de = hn::Add(de, hn::MulEven(v, F[k]));
            dout = hn::Add(dout, hn::MulOdd(v, F[k]));
        }
    }
    dot_e = hn::Add(dot_e0, dot_e1);
    dot_o = hn::Add(dot_o0, dot_o1);
}

// Same, for a vertical window of 8 preloaded i16 row vectors.
template <bool k6tap, class D16, class V16, class V32>
static HWY_INLINE void Dot8V(const D16, const V16 rows[8],
                             const hn::VFromD<D16> F[8],
                             V32& dot_e, V32& dot_o) {
    const hn::Repartition<int32_t, D16> d32;
    auto dot_e0 = hn::Zero(d32);
    auto dot_e1 = hn::Zero(d32);
    auto dot_o0 = hn::Zero(d32);
    auto dot_o1 = hn::Zero(d32);
    for (int k = k6tap ? 1 : 0; k < (k6tap ? 7 : 8); k++) {
        auto& de = (k & 1) ? dot_e1 : dot_e0;
        auto& dout = (k & 1) ? dot_o1 : dot_o0;
        de = hn::Add(de, hn::MulEven(rows[k], F[k]));
        dout = hn::Add(dout, hn::MulOdd(rows[k], F[k]));
    }
    dot_e = hn::Add(dot_e0, dot_e1);
    dot_o = hn::Add(dot_o0, dot_o1);
}

// kCap == 4 horizontal pass, two rows at once (p0/p1 == row + x - 3): both
// row windows are packed into one 8-lane vector per tap. Reads pixels
// [x-3, x+7], all within the range the
// C code reads. dot_e holds columns [r0c0, r0c2, r1c0, r1c2], dot_o the odd
// ones; the InterleaveLower/Upper halves are the two rows in column order.
template <bool k6tap, class D8, typename Src, class V32>
static HWY_INLINE void Dot8H2(const D8 d, const Src *const p0,
                              const Src *const p1, const hn::VFromD<D8> F[8],
                              V32& dot_e, V32& dot_o) {
    const auto wa = LoadI16<0>(d, p0, 8);
    const auto ha = LoadI16<0>(d, p0 + 8, k6tap ? 2 : 3);
    const auto wb = LoadI16<0>(d, p1, 8);
    const auto hb = LoadI16<0>(d, p1 + 8, k6tap ? 2 : 3);
    const hn::Repartition<int32_t, D8> d32;
    auto e0 = hn::Zero(d32);
    auto e1 = hn::Zero(d32);
    auto o0 = hn::Zero(d32);
    auto o1 = hn::Zero(d32);
    ForEachTap<k6tap ? 1 : 0, k6tap ? 7 : 8>([&](auto kc) {
        constexpr int k = decltype(kc)::value;
        const auto v = hn::ConcatLowerLower(d, ShiftedWindow<k>(d, hb, wb),
                                            ShiftedWindow<k>(d, ha, wa));
        auto& de = (k & 1) ? e1 : e0;
        auto& dout = (k & 1) ? o1 : o0;
        de = hn::Add(de, hn::MulEven(v, F[k]));
        dout = hn::Add(dout, hn::MulOdd(v, F[k]));
    });
    dot_e = hn::Add(e0, e1);
    dot_o = hn::Add(o0, o1);
}

// Rounds (dot + rnd) >> sh and stores one 4-column paired-row half as i16.
template <class V32>
static HWY_INLINE void StorePackI16x4(V32 dot, const int rnd, const int sh,
                                      int16_t *const p) {
    const hn::DFromV<V32> d32;
    const hn::Rebind<int16_t, decltype(d32)> dh;
    dot = hn::ShiftRightSame(hn::Add(dot, hn::Set(d32, rnd)), sh);
    hn::StoreN(hn::DemoteTo(dh, dot), dh, p, 4);
}

// Same, as pixels clipped to [0, bitdepth_max] (the saturating demotes are
// the clip; see StorePackPx).
template <typename Pixel, class V32>
static HWY_INLINE void StorePackPx4(V32 dot, const int rnd, const int sh,
                                    Pixel *const p, const int bitdepth_max) {
    const hn::DFromV<V32> d32;
    const hn::Rebind<Pixel, decltype(d32)> dp;
    dot = hn::ShiftRightSame(hn::Add(dot, hn::Set(d32, rnd)), sh);
    if constexpr (sizeof(Pixel) == 1) {
        const hn::Rebind<int16_t, decltype(d32)> dh;
        hn::StoreN(hn::DemoteTo(dp, hn::DemoteTo(dh, dot)), dp, p, 4);
    } else {
        hn::StoreN(hn::DemoteTo(dp, hn::Min(dot, hn::Set(d32, bitdepth_max))),
                   dp, p, 4);
    }
}

// Bilinear dot on preloaded MID rows: 16*s0 + mxy*(s1 - s0), returned as i32
// even/odd pairs. The unrounded value is (16-mxy)*s0 + mxy*s1, up to 16*7905
// for 8bpc (the horizontal pass does not shift for 8bpc/10bpc), so unlike the
// on-pixel variant below this cannot stay in 16-bit lanes; the i32
// MulEven/MulOdd accumulation is exact. v16/vm are the broadcast constants
// 16 and mxy.
template <class D16, class V16, class V32>
static HWY_INLINE void DotBilinV(const D16, const V16 s0, const V16 s1,
                                 const V16 v16, const V16 vm, V32& dot_e,
                                 V32& dot_o) {
    const auto diff = hn::Sub(s1, s0); // exact in i16: |s1 - s0| <= 4095
    dot_e = hn::Add(hn::MulEven(s0, v16), hn::MulEven(diff, vm));
    dot_o = hn::Add(hn::MulOdd(s0, v16), hn::MulOdd(diff, vm));
}

// One chunk of n columns of the bilinear filter along stride, on PIXELS.
// kI16 (8bpc) uses the (16-mxy)*s0 + mxy*s1 form in pure i16 lanes: every
// product is non-negative and the sum is <= 16*255, so no widening is needed.
// (10bpc would also fit, <= 16*1023, but is left on the i32 path to keep the
// dispatch compile-time; 12bpc pixels do not fit.) v16/vm/v16m are the
// broadcast constants 16, mxy and 16 - mxy.
template <int kCap, bool kI16, class D16, typename Src>
static HWY_INLINE hn::VFromD<D16> BilinDot(const D16 d, const Src *const p,
                                           const ptrdiff_t stride,
                                           const size_t n,
                                           const hn::VFromD<D16> v16,
                                           const hn::VFromD<D16> vm,
                                           const hn::VFromD<D16> v16m,
                                           const int rnd, const int sh) {
    const auto s0 = LoadI16<kCap>(d, p, n);
    const auto s1 = LoadI16<kCap>(d, p + stride, n);
    if constexpr (kI16) {
        const auto dot = hn::Add(hn::Mul(s0, v16m), hn::Mul(s1, vm));
        return hn::ShiftRightSame(hn::Add(dot, hn::Set(d, rnd)), sh);
    } else {
        const auto diff = hn::Sub(s1, s0); // exact in i16: |s1 - s0| <= 4095
        const auto e = hn::Add(hn::MulEven(s0, v16), hn::MulEven(diff, vm));
        const auto o = hn::Add(hn::MulOdd(s0, v16), hn::MulOdd(diff, vm));
        return RoundPack(d, e, o, rnd, sh);
    }
}

// True when the vanishing outer taps can be skipped: the regular and smooth
// filter tables have F[0] == F[7] == 0 for every subpel position.
static HWY_INLINE bool tap_6(const int8_t *const F) {
    return F[0] == 0 && F[7] == 0;
}

// Horizontal pass of the 8-tap filter shared by put/prep: filters the h + 7
// rows starting at src - 3 * ss into the i16 mid buffer (stride ms) as
// (dot + rnd_h) >> (6 - ib); the values fit int16 (see CombineEvenOdd).
template <typename Pixel, int kCap, bool k6tap, class D16>
static void hwy_8tap_h_pass(const D16 d, const int8_t *const fh,
                            const hn::VFromD<D16> (&hf)[8],
                            const Pixel *const src, const ptrdiff_t ss,
                            const int w, const int h, const int ib,
                            int16_t *const mid, const int ms) {
    using DT32 = hn::Repartition<int32_t, D16>;
    const int N = (int) hn::Lanes(d);
    const int rnd_h = (1 << (6 - ib)) >> 1;
    if constexpr (kCap == 4) {
        // Two rows per 8-lane vector, see Dot8H2.
        const hn::CappedTag<int16_t, 8> d8;
        hn::VFromD<decltype(d8)> hf8[8];
        LoadTaps(d8, fh, hf8);
        const hn::Repartition<int32_t, decltype(d8)> d32w;
        const Pixel *srow = src - 3 * ss;
        int y = 0;
        for (; y + 2 <= h + 7; y += 2, srow += 2 * ss) {
            hn::VFromD<decltype(d32w)> dot_e, dot_o;
            Dot8H2<k6tap>(d8, srow - 3, srow + ss - 3, hf8, dot_e, dot_o);
            StorePackI16x4(hn::InterleaveLower(dot_e, dot_o), rnd_h,
                           6 - ib, mid + y * ms);
            StorePackI16x4(hn::InterleaveUpper(d32w, dot_e, dot_o),
                           rnd_h, 6 - ib, mid + (y + 1) * ms);
        }
        if (y < h + 7) {
            hn::VFromD<decltype(d32w)> dot_e, dot_o;
            Dot8H2<k6tap>(d8, srow - 3, srow - 3, hf8, dot_e, dot_o);
            StorePackI16x4(hn::InterleaveLower(dot_e, dot_o), rnd_h,
                           6 - ib, mid + y * ms);
        }
    } else if constexpr (kCap == 8) {
        const Pixel *srow = src - 3 * ss;
        int y = 0;
        for (; y + 2 <= h + 7; y += 2, srow += 2 * ss)
            for (int x = 0; x < w; x += N) {
                const size_t n = (size_t) (w - x < N ? w - x : N);
                hn::VFromD<DT32> dot_e, dot_o;
                Dot8H<kCap, k6tap>(d, srow + x - 3, n, hf, dot_e, dot_o);
                StorePackI16<kCap>(d, mid + y * ms + x, n,
                                   dot_e, dot_o, rnd_h, 6 - ib);
                Dot8H<kCap, k6tap>(d, srow + ss + x - 3, n, hf, dot_e, dot_o);
                StorePackI16<kCap>(d, mid + (y + 1) * ms + x, n,
                                   dot_e, dot_o, rnd_h, 6 - ib);
            }
        for (; y < h + 7; y++, srow += ss)
            for (int x = 0; x < w; x += N) {
                const size_t n = (size_t) (w - x < N ? w - x : N);
                hn::VFromD<DT32> dot_e, dot_o;
                Dot8H<kCap, k6tap>(d, srow + x - 3, n, hf, dot_e, dot_o);
                StorePackI16<kCap>(d, mid + y * ms + x, n,
                                   dot_e, dot_o, rnd_h, 6 - ib);
            }
    } else {
        const Pixel *srow = src - 3 * ss;
        for (int y = 0; y < h + 7; y++, srow += ss)
            for (int x = 0; x < w; x += N) {
                const size_t n = (size_t) (w - x < N ? w - x : N);
                hn::VFromD<DT32> dot_e, dot_o;
                Dot8H<kCap, k6tap>(d, srow + x - 3, n, hf, dot_e, dot_o);
                StorePackI16<kCap>(d, mid + y * ms + x, n,
                                   dot_e, dot_o, rnd_h, 6 - ib);
            }
    }
}

// Vertical pass over the h + 7-row mid buffer, shared by put/prep: Dot8V
// over the sliding 8/9-row window per column block; store(orow, dot_e,
// dot_o, n) rounds/clips/writes one output row, orow advances os per row.
template <int kCap, bool k6tap, class D16, typename Out, class Store>
static HWY_INLINE void hwy_8tap_v_pass(const D16 d, const int16_t *const mid,
                                       Out *const out, const ptrdiff_t os,
                                       const int w, const int h,
                                       const hn::VFromD<D16> (&vf)[8],
                                       const Store& store) {
    using DT32 = hn::Repartition<int32_t, D16>;
    const int N = (int) hn::Lanes(d);
    for (int x = 0; x < w; x += N) {
        const size_t n = (size_t) (w - x < N ? w - x : N);
        const int16_t *mrow = mid + x;
        Out *orow = out + x;
        int y = 0;
        // Two output rows per iteration: they share 7 of 9 inputs.
        for (; y + 2 <= h; y += 2, mrow += 2 * w, orow += 2 * os) {
            hn::VFromD<D16> rows[9];
            for (int k = 0; k < 9; k++)
                rows[k] = LoadI16<kCap>(d, mrow + k * w, n);
            hn::VFromD<DT32> dot_e, dot_o;
            Dot8V<k6tap>(d, rows, vf, dot_e, dot_o);
            store(orow, dot_e, dot_o, n);
            Dot8V<k6tap>(d, rows + 1, vf, dot_e, dot_o);
            store(orow + os, dot_e, dot_o, n);
        }
        for (; y < h; y++, mrow += w, orow += os) {
            hn::VFromD<D16> rows[8];
            for (int k = 0; k < 8; k++)
                rows[k] = LoadI16<kCap>(d, mrow + k * w, n);
            hn::VFromD<DT32> dot_e, dot_o;
            Dot8V<k6tap>(d, rows, vf, dot_e, dot_o);
            store(orow, dot_e, dot_o, n);
        }
    }
}

// Same loop structure with the filter window sliding over the source rows
// (my == 0 in put/prep), so the loads are pixels along ss.
template <typename Pixel, int kCap, bool k6tap, class D16, typename Out,
          class Store>
static HWY_INLINE void hwy_8tap_v_only(const D16 d, const Pixel *const src,
                                       const ptrdiff_t ss, Out *const out,
                                       const ptrdiff_t os, const int w,
                                       const int h,
                                       const hn::VFromD<D16> (&vf)[8],
                                       const Store& store) {
    using DT32 = hn::Repartition<int32_t, D16>;
    const int N = (int) hn::Lanes(d);
    for (int x = 0; x < w; x += N) {
        const size_t n = (size_t) (w - x < N ? w - x : N);
        const Pixel *srow = src + x - 3 * ss;
        Out *orow = out + x;
        int y = 0;
        // Two output rows per iteration: they share 7 of 9 inputs.
        for (; y + 2 <= h; y += 2, srow += 2 * ss, orow += 2 * os) {
            hn::VFromD<D16> rows[9];
            for (int k = 0; k < 9; k++)
                rows[k] = LoadI16<kCap>(d, srow + k * ss, n);
            hn::VFromD<DT32> dot_e, dot_o;
            Dot8V<k6tap>(d, rows, vf, dot_e, dot_o);
            store(orow, dot_e, dot_o, n);
            Dot8V<k6tap>(d, rows + 1, vf, dot_e, dot_o);
            store(orow + os, dot_e, dot_o, n);
        }
        for (; y < h; y++, srow += ss, orow += os) {
            hn::VFromD<D16> rows[8];
            for (int k = 0; k < 8; k++)
                rows[k] = LoadI16<kCap>(d, srow + k * ss, n);
            hn::VFromD<DT32> dot_e, dot_o;
            Dot8V<k6tap>(d, rows, vf, dot_e, dot_o);
            store(orow, dot_e, dot_o, n);
        }
    }
}

// Phase 0 of the scaled horizontal pass takes the unfiltered path in C
// (src[ioff] << intermediate_bits); a delta tap row reproduces it exactly
// through the dot path: (64 * px + ((1 << (6 - ib)) >> 1)) >> (6 - ib) ==
// px << ib.
alignas(8) const int8_t kDeltaTaps[8] = { 0, 0, 0, 64, 0, 0, 0, 0 };

// 8x8 i16 transpose via three levels of interleaves (i16, then i32, then
// i64 lanes).
template <class D16>
static HWY_INLINE void Transpose8x8I16(const D16 d,
                                       const hn::VFromD<D16> in[8],
                                       hn::VFromD<D16> out[8]) {
    const hn::Repartition<int32_t, D16> d32;
    const hn::Repartition<int64_t, D16> d64;
    hn::VFromD<D16> t[8], u[8];
    for (int i = 0; i < 4; i++) {
        t[2 * i + 0] = hn::InterleaveLower(in[2 * i], in[2 * i + 1]);
        t[2 * i + 1] = hn::InterleaveUpper(d, in[2 * i], in[2 * i + 1]);
    }
    for (int i = 0; i < 2; i++) {
        u[i + 0] = hn::BitCast(d, hn::InterleaveLower(
            d32, hn::BitCast(d32, t[i]), hn::BitCast(d32, t[i + 2])));
        u[i + 2] = hn::BitCast(d, hn::InterleaveLower(
            d32, hn::BitCast(d32, t[i + 4]), hn::BitCast(d32, t[i + 6])));
        u[i + 4] = hn::BitCast(d, hn::InterleaveUpper(
            d32, hn::BitCast(d32, t[i]), hn::BitCast(d32, t[i + 2])));
        u[i + 6] = hn::BitCast(d, hn::InterleaveUpper(
            d32, hn::BitCast(d32, t[i + 4]), hn::BitCast(d32, t[i + 6])));
    }
    // u[] lanes: u0 = cols 0/1 rows 0-3, u2 = cols 0/1 rows 4-7, u1/u3 the
    // same for cols 4/5, u4..u7 for cols 2/3 and 6/7.
    using V64 = hn::VFromD<decltype(d64)>;
    const V64 u0 = hn::BitCast(d64, u[0]);
    const V64 u1 = hn::BitCast(d64, u[1]);
    const V64 u2 = hn::BitCast(d64, u[2]);
    const V64 u3 = hn::BitCast(d64, u[3]);
    const V64 u4 = hn::BitCast(d64, u[4]);
    const V64 u5 = hn::BitCast(d64, u[5]);
    const V64 u6 = hn::BitCast(d64, u[6]);
    const V64 u7 = hn::BitCast(d64, u[7]);
    out[0] = hn::BitCast(d, hn::InterleaveLower(d64, u0, u2));
    out[1] = hn::BitCast(d, hn::InterleaveUpper(d64, u0, u2));
    out[2] = hn::BitCast(d, hn::InterleaveLower(d64, u4, u6));
    out[3] = hn::BitCast(d, hn::InterleaveUpper(d64, u4, u6));
    out[4] = hn::BitCast(d, hn::InterleaveLower(d64, u1, u3));
    out[5] = hn::BitCast(d, hn::InterleaveUpper(d64, u1, u3));
    out[6] = hn::BitCast(d, hn::InterleaveLower(d64, u5, u7));
    out[7] = hn::BitCast(d, hn::InterleaveUpper(d64, u5, u7));
}

// Transposed tap matrix for per-column filters: ft[k] holds tap k of the 8
// column filters, so a per-column dot is MulEven/MulOdd(v, ft[k]) with the
// column index as the lane. rows are 8-byte filter rows (the tables in
// src/tables.c, or kDeltaTaps).
template <class D16>
static HWY_INLINE void LoadTapsT(const D16 d, const int8_t *const rows[8],
                                 hn::VFromD<D16> ft[8]) {
    const hn::CappedTag<int8_t, 8> d8;
    hn::VFromD<D16> r[8];
    for (int j = 0; j < 8; j++)
        r[j] = hn::PromoteTo(d, hn::LoadU(d8, rows[j]));
    Transpose8x8I16(d, r, ft);
}

// Source offset (ioff) and filter phase (imx >> 6) of every output column of
// the scaled kernels, from the C accumulation imx += dx; ioff += imx >> 10;
// imx &= 0x3ff. Identical on every horizontally filtered row of a call.
static HWY_INLINE void ScaledHOffsets(const int w, const int mx, const int dx,
                                      int32_t *const ioff,
                                      uint8_t *const phase) {
    int imx = mx, io = 0;
    for (int x = 0; x < w; x++) {
        phase[x] = (uint8_t) (imx >> 6);
        ioff[x] = io;
        imx += dx;
        io += imx >> 10;
        imx &= 0x3ff;
    }
}

// Per-call horizontal geometry of the 8-tap scaled kernels: per 8-column
// chunk, the window offsets and the transposed per-column tap matrix. Lanes
// past w repeat the last real output column, keeping the window loads inside
// the range the C code reads; their results land in unused mid columns.
// Returns the chunk count.
template <class D16>
static HWY_INLINE int ScaledHInit(const D16 d, const int w, const int mx,
                                  const int dx, const int filter_type,
                                  int32_t off[16][8],
                                  hn::VFromD<D16> taps[16][8]) {
    const int table = w > 4 ? (filter_type & 3) : (3 + (filter_type & 1));
    int32_t ioff[128];
    uint8_t phase[128];
    ScaledHOffsets(w, mx, dx, ioff, phase);
    const int n = (w + 7) >> 3;
    for (int c = 0; c < n; c++) {
        const int8_t *rows[8];
        for (int j = 0; j < 8; j++) {
            const int x = 8 * c + j < w ? 8 * c + j : w - 1;
            off[c][j] = ioff[x];
            rows[j] = phase[x] ? dav1d_mc_subpel_filters[table][phase[x] - 1]
                               : kDeltaTaps;
        }
        LoadTapsT(d, rows, taps[c]);
    }
    return n;
}

// Same, for the bilinear scaled kernels: per-lane constant pairs holding
// interleaved 16 - m / m multipliers (m = imx >> 6).
template <class D16>
static HWY_INLINE int ScaledHInitBilin(const D16 d, const int w, const int mx,
                                       const int dx, int32_t off[16][8],
                                       hn::VFromD<D16> cm[16][2]) {
    int32_t ioff[128];
    uint8_t phase[128];
    ScaledHOffsets(w, mx, dx, ioff, phase);
    const int n = (w + 7) >> 3;
    for (int c = 0; c < n; c++) {
        alignas(16) int16_t cb[16];
        for (int j = 0; j < 8; j++) {
            const int x = 8 * c + j < w ? 8 * c + j : w - 1;
            off[c][j] = ioff[x];
            cb[2 * j] = 16 - phase[x];
            cb[2 * j + 1] = phase[x];
        }
        cm[c][0] = hn::Load(d, cb);
        cm[c][1] = hn::Load(d, cb + 8);
    }
    return n;
}

// One horizontally filtered row of the 8-tap scaled kernels into a mid row:
// (sum_k taps[k][col] * srow[off(col) - 3 + k] + rnd) >> sh. Each window load
// reads srow[off - 3 .. off + 4], exactly the C footprint of that output.
template <class D16, typename Pixel>
static HWY_INLINE void ScaledHRow8tap(const D16 d, const Pixel *const srow,
                                      const int32_t off[16][8],
                                      const hn::VFromD<D16> taps[16][8],
                                      const int nch, int16_t *const mrow,
                                      const int rnd, const int sh) {
    const hn::Repartition<int32_t, D16> d32;
    for (int c = 0; c < nch; c++) {
        hn::VFromD<D16> w8[8], mt[8];
        for (int j = 0; j < 8; j++)
            w8[j] = LoadI16<8>(d, srow + off[c][j] - 3, 8);
        Transpose8x8I16(d, w8, mt);
        auto de0 = hn::Zero(d32), de1 = hn::Zero(d32);
        auto do0 = hn::Zero(d32), do1 = hn::Zero(d32);
        for (int k = 0; k < 8; k++) {
            auto& de = (k & 1) ? de1 : de0;
            auto& dout = (k & 1) ? do1 : do0;
            de = hn::Add(de, hn::MulEven(mt[k], taps[c][k]));
            dout = hn::Add(dout, hn::MulOdd(mt[k], taps[c][k]));
        }
        StoreI16<8>(d, mrow + 8 * c, 8,
                    RoundPack(d, hn::Add(de0, de1), hn::Add(do0, do1), rnd, sh));
    }
}

// Same, bilinear: per output, the exact C footprint (src[ioff], src[ioff+1])
// is read as one packed 2-pixel scalar load, so lanes alternate s0/s1 and
// (16 - m) * s0 + m * s1 == MulEven(v, cm) + MulOdd(v, cm) with cm holding
// interleaved 16 - m / m pairs. (16 - m) * s0 + m * s1 <= 16 * 4095.
template <class D16, typename Pixel>
static HWY_INLINE void ScaledHRowBilin(const D16 d, const Pixel *const srow,
                                       const int32_t off[16][8],
                                       const hn::VFromD<D16> cm[16][2],
                                       const int nch, int16_t *const mrow,
                                       const int rnd, const int sh) {
    using BP = typename std::conditional<sizeof(Pixel) == 1, uint16_t,
                                         uint32_t>::type;
    for (int c = 0; c < nch; c++) {
        alignas(16) BP pairs[8];
        for (int j = 0; j < 8; j++)
            memcpy(&pairs[j], srow + off[c][j], sizeof(BP));
        for (int hf = 0; hf < 2; hf++) {
            const auto v = LoadI16<8>(d, reinterpret_cast<const Pixel *>(pairs)
                                      + 8 * hf, 8);
            const auto dot = hn::Add(hn::MulEven(v, cm[c][hf]),
                                     hn::MulOdd(v, cm[c][hf]));
            StorePackI16x4(dot, rnd, sh, mrow + 8 * c + 4 * hf);
        }
    }
}

// put_c from src/mc_tmpl.c.
template <typename Pixel, int kCap>
static void hwy_put_c(Pixel *dst, const ptrdiff_t dst_stride,
                      const Pixel *src, const ptrdiff_t src_stride,
                      const int w, const int h) {
    constexpr bool kFull = kCap != 0;
    const hn::CappedTag<Pixel, kCap == 0 ? 8 : kCap> d;
    const int N = (int) hn::Lanes(d);
    for (int y = 0; y < h; y++, src += src_stride, dst += dst_stride)
        for (int x = 0; x < w; x += N) {
            const size_t n = (size_t) (w - x < N ? w - x : N);
            const auto v = kFull ? hn::LoadU(d, src + x) : hn::LoadN(d, src + x, n);
            if (kFull) hn::StoreU(v, d, dst + x); else hn::StoreN(v, d, dst + x, n);
        }
}

// prep_c from src/mc_tmpl.c: (src << intermediate_bits) - PREP_BIAS; the
// shift cannot overflow i16 (max 4095 << 2, 1023 << 4 or 255 << 4).
template <typename Pixel, int kCap>
static void hwy_prep_c(int16_t *tmp, const Pixel *src,
                       const ptrdiff_t src_stride, const int w, const int h,
                       const int intermediate_bits, const int prep_bias) {
    using DT16 = hn::CappedTag<int16_t, kCap == 0 ? 8 : kCap>;
    const DT16 d;
    const int N = (int) hn::Lanes(d);
    const auto vbias = hn::Set(d, prep_bias);
    for (int y = 0; y < h; y++, src += src_stride, tmp += w)
        for (int x = 0; x < w; x += N) {
            const size_t n = (size_t) (w - x < N ? w - x : N);
            const auto v = hn::Sub(hn::ShiftLeftSame(LoadI16<kCap>(d, src + x, n),
                                                     intermediate_bits), vbias);
            StoreI16<kCap>(d, tmp + x, n, v);
        }
}

// put_8tap_c from src/mc_tmpl.c (mx == 0 / my == 0 fast paths included).
template <typename Pixel, int kCap>
static void hwy_put_8tap_impl(Pixel *const dst, const ptrdiff_t dst_stride,
                         const Pixel *const src, const ptrdiff_t src_stride,
                         const int w, const int h, const int mx, const int my,
                         const int filter_type, const int bitdepth_max,
                         const std::integral_constant<int, kCap>) {
    const int ib = sizeof(Pixel) == 1 ? 4 : 13 - hwy_ulog2(bitdepth_max);
    const int8_t *const fh = !mx ? NULL :
        dav1d_mc_subpel_filters[w > 4 ? (filter_type & 3)
                                      : (3 + (filter_type & 1))][mx - 1];
    const int8_t *const fv = !my ? NULL :
        dav1d_mc_subpel_filters[h > 4 ? (filter_type >> 2)
                                      : (3 + ((filter_type >> 2) & 1))][my - 1];
    const ptrdiff_t ds = dst_stride / (ptrdiff_t) sizeof(Pixel);
    const ptrdiff_t ss = src_stride / (ptrdiff_t) sizeof(Pixel);
    using DT16 = hn::CappedTag<int16_t, kCap == 0 ? 8 : kCap>;
    using DT32 = hn::Repartition<int32_t, DT16>;
    const DT16 d;
    const int N = (int) hn::Lanes(d);

    if (fh) {
        hn::VFromD<DT16> hf[8]; LoadTaps(d, fh, hf);
        if (fv) {
            hn::VFromD<DT16> vf[8]; LoadTaps(d, fv, vf);
            int16_t mid[kMidStride * 135];
            if (tap_6(fh))
                hwy_8tap_h_pass<Pixel, kCap, true>(d, fh, hf, src, ss, w, h,
                                                   ib, mid, w);
            else
                hwy_8tap_h_pass<Pixel, kCap, false>(d, fh, hf, src, ss, w, h,
                                                    ib, mid, w);
            const int rnd_v = (1 << (6 + ib)) >> 1;
            const auto store_px = [&](Pixel *const p, auto dot_e, auto dot_o,
                                      const size_t n) {
                StorePackPx<kCap>(d, p, n, dot_e, dot_o, rnd_v, 6 + ib,
                                  bitdepth_max);
            };
            if (tap_6(fv))
                hwy_8tap_v_pass<kCap, true>(d, mid, dst, ds, w, h, vf,
                                            store_px);
            else
                hwy_8tap_v_pass<kCap, false>(d, mid, dst, ds, w, h, vf,
                                             store_px);
        } else {
            const int rnd = 32 + ((1 << (6 - ib)) >> 1);
            const auto h_only = [&](auto k6) {
                constexpr bool k6v = decltype(k6)::value;
                if constexpr (kCap == 4) {
                    // Two rows per 8-lane vector, see Dot8H2.
                    const hn::CappedTag<int16_t, 8> d8;
                    hn::VFromD<decltype(d8)> hf8[8];
                    LoadTaps(d8, fh, hf8);
                    const hn::Repartition<int32_t, decltype(d8)> d32w;
                    const Pixel *srow = src;
                    Pixel *drow = dst;
                    int y = 0;
                    for (; y + 2 <= h; y += 2, srow += 2 * ss, drow += 2 * ds) {
                        hn::VFromD<decltype(d32w)> dot_e, dot_o;
                        Dot8H2<k6v>(d8, srow - 3, srow + ss - 3, hf8, dot_e, dot_o);
                        StorePackPx4(hn::InterleaveLower(dot_e, dot_o), rnd, 6,
                                     drow, bitdepth_max);
                        StorePackPx4(hn::InterleaveUpper(d32w, dot_e, dot_o),
                                     rnd, 6, drow + ds, bitdepth_max);
                    }
                    if (y < h) {
                        hn::VFromD<decltype(d32w)> dot_e, dot_o;
                        Dot8H2<k6v>(d8, srow - 3, srow - 3, hf8, dot_e, dot_o);
                        StorePackPx4(hn::InterleaveLower(dot_e, dot_o), rnd, 6,
                                     drow, bitdepth_max);
                    }
                } else {
                    const Pixel *srow = src;
                    Pixel *drow = dst;
                    for (int y = 0; y < h; y++, srow += ss, drow += ds)
                        for (int x = 0; x < w; x += N) {
                            const size_t n = (size_t) (w - x < N ? w - x : N);
                            hn::VFromD<DT32> dot_e, dot_o;
                            Dot8H<kCap, k6v>(d, srow + x - 3, n, hf, dot_e, dot_o);
                            StorePackPx<kCap>(d, drow + x, n, dot_e, dot_o, rnd, 6,
                                              bitdepth_max);
                        }
                }
            };
            if (tap_6(fh)) h_only(std::true_type{});
            else h_only(std::false_type{});
        }
    } else if (fv) {
        hn::VFromD<DT16> vf[8]; LoadTaps(d, fv, vf);
        const auto store_px = [&](Pixel *const p, auto dot_e, auto dot_o,
                                  const size_t n) {
            StorePackPx<kCap>(d, p, n, dot_e, dot_o, 32, 6, bitdepth_max);
        };
        if (tap_6(fv))
            hwy_8tap_v_only<Pixel, kCap, true>(d, src, ss, dst, ds, w, h, vf,
                                               store_px);
        else
            hwy_8tap_v_only<Pixel, kCap, false>(d, src, ss, dst, ds, w, h, vf,
                                                store_px);
    } else
        hwy_put_c<Pixel, kCap>(dst, ds, src, ss, w, h);
}

// prep_8tap_c from src/mc_tmpl.c.
template <typename Pixel, int kCap>
static void hwy_prep_8tap_impl(int16_t *tmp, const Pixel *const src,
                          const ptrdiff_t src_stride, const int w, const int h,
                          const int mx, const int my, const int filter_type,
                          const int bitdepth_max,
                         const std::integral_constant<int, kCap>) {
    const int ib = sizeof(Pixel) == 1 ? 4 : 13 - hwy_ulog2(bitdepth_max);
    const int prep_bias = sizeof(Pixel) == 1 ? 0 : 8192;
    const int8_t *const fh = !mx ? NULL :
        dav1d_mc_subpel_filters[w > 4 ? (filter_type & 3)
                                      : (3 + (filter_type & 1))][mx - 1];
    const int8_t *const fv = !my ? NULL :
        dav1d_mc_subpel_filters[h > 4 ? (filter_type >> 2)
                                      : (3 + ((filter_type >> 2) & 1))][my - 1];
    const ptrdiff_t ss = src_stride / (ptrdiff_t) sizeof(Pixel);
    using DT16 = hn::CappedTag<int16_t, kCap == 0 ? 8 : kCap>;
    using DT32 = hn::Repartition<int32_t, DT16>;
    const DT16 d;
    const int N = (int) hn::Lanes(d);

    if (fh) {
        hn::VFromD<DT16> hf[8]; LoadTaps(d, fh, hf);
        if (fv) {
            hn::VFromD<DT16> vf[8]; LoadTaps(d, fv, vf);
            int16_t mid[kMidStride * 135];
            if (tap_6(fh))
                hwy_8tap_h_pass<Pixel, kCap, true>(d, fh, hf, src, ss, w, h,
                                                   ib, mid, w);
            else
                hwy_8tap_h_pass<Pixel, kCap, false>(d, fh, hf, src, ss, w, h,
                                                    ib, mid, w);
            // (dot + 32) >> 6 - PREP_BIAS, computed as
            // (dot + 32 - (PREP_BIAS << 6)) >> 6 (equal for arithmetic
            // shifts): the pre-bias value can exceed int16 (up to 36985), so
            // the bias must be folded in before the i32 -> i16 pack, not
            // subtracted after it.
            const auto store_i16 = [&](int16_t *const p, auto dot_e,
                                       auto dot_o, const size_t n) {
                StorePackI16<kCap>(d, p, n, dot_e, dot_o,
                                   32 - (prep_bias << 6), 6);
            };
            if (tap_6(fv))
                hwy_8tap_v_pass<kCap, true>(d, mid, tmp, w, w, h, vf,
                                            store_i16);
            else
                hwy_8tap_v_pass<kCap, false>(d, mid, tmp, w, w, h, vf,
                                             store_i16);
        } else {
            const auto h_only = [&](auto k6) {
                constexpr bool k6v = decltype(k6)::value;
                // PREP_BIAS folded into the rounding constant (exact for
                // arithmetic shifts; result fits int16).
                const int rnd = ((1 << (6 - ib)) >> 1) - (prep_bias << (6 - ib));
                if constexpr (kCap == 4) {
                    // Two rows per 8-lane vector, see Dot8H2.
                    const hn::CappedTag<int16_t, 8> d8;
                    hn::VFromD<decltype(d8)> hf8[8];
                    LoadTaps(d8, fh, hf8);
                    const hn::Repartition<int32_t, decltype(d8)> d32w;
                    const Pixel *srow = src;
                    int16_t *trow = tmp;
                    int y = 0;
                    for (; y + 2 <= h; y += 2, srow += 2 * ss, trow += 2 * w) {
                        hn::VFromD<decltype(d32w)> dot_e, dot_o;
                        Dot8H2<k6v>(d8, srow - 3, srow + ss - 3, hf8, dot_e, dot_o);
                        StorePackI16x4(hn::InterleaveLower(dot_e, dot_o), rnd,
                                       6 - ib, trow);
                        StorePackI16x4(hn::InterleaveUpper(d32w, dot_e, dot_o),
                                       rnd, 6 - ib, trow + w);
                    }
                    if (y < h) {
                        hn::VFromD<decltype(d32w)> dot_e, dot_o;
                        Dot8H2<k6v>(d8, srow - 3, srow - 3, hf8, dot_e, dot_o);
                        StorePackI16x4(hn::InterleaveLower(dot_e, dot_o), rnd,
                                       6 - ib, trow);
                    }
                } else {
                    const Pixel *srow = src;
                    int16_t *trow = tmp;
                    for (int y = 0; y < h; y++, srow += ss, trow += w)
                        for (int x = 0; x < w; x += N) {
                            const size_t n = (size_t) (w - x < N ? w - x : N);
                            hn::VFromD<DT32> dot_e, dot_o;
                            Dot8H<kCap, k6v>(d, srow + x - 3, n, hf, dot_e, dot_o);
                            StorePackI16<kCap>(d, trow + x, n, dot_e, dot_o,
                                               rnd, 6 - ib);
                        }
                }
            };
            if (tap_6(fh)) h_only(std::true_type{});
            else h_only(std::false_type{});
        }
    } else if (fv) {
        hn::VFromD<DT16> vf[8]; LoadTaps(d, fv, vf);
        const int rnd = (1 << (6 - ib)) >> 1;
        const auto store_i16 = [&](int16_t *const p, auto dot_e, auto dot_o,
                                   const size_t n) {
            StorePackI16<kCap>(d, p, n, dot_e, dot_o,
                               rnd - (prep_bias << (6 - ib)), 6 - ib);
        };
        if (tap_6(fv))
            hwy_8tap_v_only<Pixel, kCap, true>(d, src, ss, tmp, w, w, h, vf,
                                               store_i16);
        else
            hwy_8tap_v_only<Pixel, kCap, false>(d, src, ss, tmp, w, w, h, vf,
                                                store_i16);
    } else
        hwy_prep_c<Pixel, kCap>(tmp, src, ss, w, h, ib, prep_bias);
}

// put_bilin_c from src/mc_tmpl.c.
template <typename Pixel, int kCap>
static void hwy_put_bilin_impl(Pixel *const dst, const ptrdiff_t dst_stride,
                          const Pixel *const src, const ptrdiff_t src_stride,
                          const int w, const int h, const int mx, const int my,
                          const int bitdepth_max,
                         const std::integral_constant<int, kCap>) {
    const int ib = sizeof(Pixel) == 1 ? 4 : 13 - hwy_ulog2(bitdepth_max);
    const ptrdiff_t ds = dst_stride / (ptrdiff_t) sizeof(Pixel);
    const ptrdiff_t ss = src_stride / (ptrdiff_t) sizeof(Pixel);
    using DT16 = hn::CappedTag<int16_t, kCap == 0 ? 8 : kCap>;
    using DT32 = hn::Repartition<int32_t, DT16>;
    const DT16 d;
    const int N = (int) hn::Lanes(d);
    const auto vzero = hn::Zero(d);
    const auto vmax = hn::Set(d, bitdepth_max);
    const auto v16 = hn::Set(d, 16);
    // Pure-i16 pixel-domain path (see BilinDot): always valid for 8bpc, and
    // for 10-bit content at runtime (16*1023 fits); 12-bit uses the i32 path.
    const bool px_i16 = sizeof(Pixel) == 1 || bitdepth_max <= 0x3ff;

    const auto run = [&](auto kI16) {
        constexpr bool kI16v = decltype(kI16)::value;
    if (mx) {
        const auto vmx = hn::Set(d, mx);
        const auto vmx16 = hn::Set(d, 16 - mx);
        if (my) {
            const auto vmy = hn::Set(d, my);
            int16_t mid[kMidStride * 129];
            const Pixel *srow = src;
            const int rnd_h = (1 << (4 - ib)) >> 1;
            for (int y = 0; y < h + 1; y++, srow += ss)
                for (int x = 0; x < w; x += N) {
                    const size_t n = (size_t) (w - x < N ? w - x : N);
                    const auto v = BilinDot<kCap, kI16v>(d, srow + x, 1, n, v16,
                                                        vmx, vmx16, rnd_h, 4 - ib);
                    StoreI16<kCap>(d, mid + y * kMidStride + x, n, v);
                }
            const int rnd_v = (1 << (4 + ib)) >> 1;
            for (int x = 0; x < w; x += N) {
                const size_t n = (size_t) (w - x < N ? w - x : N);
                auto m0 = LoadI16<kCap>(d, mid + x, n);
                Pixel *drow = dst + x;
                for (int y = 0; y < h; y++, drow += ds) {
                    const auto m1 = LoadI16<kCap>(d, mid + (y + 1) * kMidStride + x, n);
                    hn::VFromD<DT32> dot_e, dot_o;
                    DotBilinV(d, m0, m1, v16, vmy, dot_e, dot_o);
                    StorePackPx<kCap>(d, drow, n, dot_e, dot_o, rnd_v, 4 + ib,
                                      bitdepth_max);
                    m0 = m1;
                }
            }
        } else {
            // Two sequential rounding steps, as in the C code. The first
            // result is non-negative, so the second can use i16 shifts.
            const int rnd_h = (1 << (4 - ib)) >> 1;
            const int rnd2 = (1 << ib) >> 1;
            const auto vrnd2 = hn::Set(d, rnd2);
            const Pixel *srow = src;
            Pixel *drow = dst;
            for (int y = 0; y < h; y++, srow += ss, drow += ds)
                for (int x = 0; x < w; x += N) {
                    const size_t n = (size_t) (w - x < N ? w - x : N);
                    const auto px = BilinDot<kCap, kI16v>(d, srow + x, 1, n, v16,
                                                         vmx, vmx16, rnd_h, 4 - ib);
                    const auto v = hn::ShiftRightSame(hn::Add(px, vrnd2), ib);
                    StorePx<kCap>(d, drow + x, n, hn::Clamp(v, vzero, vmax));
                }
        }
    } else if (my) {
        const auto vmy = hn::Set(d, my);
        const auto vmy16 = hn::Set(d, 16 - my);
        const Pixel *srow = src;
        Pixel *drow = dst;
        for (int y = 0; y < h; y++, srow += ss, drow += ds)
            for (int x = 0; x < w; x += N) {
                const size_t n = (size_t) (w - x < N ? w - x : N);
                const auto v = BilinDot<kCap, kI16v>(d, srow + x, ss, n, v16,
                                                    vmy, vmy16, 8, 4);
                StorePx<kCap>(d, drow + x, n, hn::Clamp(v, vzero, vmax));
            }
    } else
        hwy_put_c<Pixel, kCap>(dst, ds, src, ss, w, h);
    };
    if (px_i16) run(std::true_type{});
    else run(std::false_type{});
}

// prep_bilin_c from src/mc_tmpl.c.
template <typename Pixel, int kCap>
static void hwy_prep_bilin_impl(int16_t *tmp, const Pixel *const src,
                           const ptrdiff_t src_stride, const int w, const int h,
                           const int mx, const int my, const int bitdepth_max,
                         const std::integral_constant<int, kCap>) {
    const int ib = sizeof(Pixel) == 1 ? 4 : 13 - hwy_ulog2(bitdepth_max);
    const int prep_bias = sizeof(Pixel) == 1 ? 0 : 8192;
    const ptrdiff_t ss = src_stride / (ptrdiff_t) sizeof(Pixel);
    using DT16 = hn::CappedTag<int16_t, kCap == 0 ? 8 : kCap>;
    using DT32 = hn::Repartition<int32_t, DT16>;
    const DT16 d;
    const int N = (int) hn::Lanes(d);
    const auto vbias = hn::Set(d, prep_bias);
    const auto v16 = hn::Set(d, 16);
    const bool px_i16 = sizeof(Pixel) == 1 || bitdepth_max <= 0x3ff;

    const auto run = [&](auto kI16) {
        constexpr bool kI16v = decltype(kI16)::value;
    if (mx) {
        const auto vmx = hn::Set(d, mx);
        const auto vmx16 = hn::Set(d, 16 - mx);
        if (my) {
            const auto vmy = hn::Set(d, my);
            int16_t mid[kMidStride * 129];
            const Pixel *srow = src;
            const int rnd_h = (1 << (4 - ib)) >> 1;
            for (int y = 0; y < h + 1; y++, srow += ss)
                for (int x = 0; x < w; x += N) {
                    const size_t n = (size_t) (w - x < N ? w - x : N);
                    const auto v = BilinDot<kCap, kI16v>(d, srow + x, 1, n, v16,
                                                        vmx, vmx16, rnd_h, 4 - ib);
                    StoreI16<kCap>(d, mid + y * kMidStride + x, n, v);
                }
            for (int x = 0; x < w; x += N) {
                const size_t n = (size_t) (w - x < N ? w - x : N);
                auto m0 = LoadI16<kCap>(d, mid + x, n);
                int16_t *trow = tmp + x;
                for (int y = 0; y < h; y++, trow += w) {
                    const auto m1 = LoadI16<kCap>(d, mid + (y + 1) * kMidStride + x, n);
                    hn::VFromD<DT32> dot_e, dot_o;
                    DotBilinV(d, m0, m1, v16, vmy, dot_e, dot_o);
                    // PREP_BIAS folded into the rounding constant (exact).
                    StorePackI16<kCap>(d, trow, n, dot_e, dot_o,
                                       8 - (prep_bias << 4), 4);
                    m0 = m1;
                }
            }
        } else {
            const int rnd = (1 << (4 - ib)) >> 1;
            const Pixel *srow = src;
            for (int y = 0; y < h; y++, srow += ss, tmp += w)
                for (int x = 0; x < w; x += N) {
                    const size_t n = (size_t) (w - x < N ? w - x : N);
                    const auto v = BilinDot<kCap, kI16v>(d, srow + x, 1, n, v16,
                                                        vmx, vmx16, rnd, 4 - ib);
                    StoreI16<kCap>(d, tmp + x, n, hn::Sub(v, vbias));
                }
        }
    } else if (my) {
        const auto vmy = hn::Set(d, my);
        const auto vmy16 = hn::Set(d, 16 - my);
        const int rnd = (1 << (4 - ib)) >> 1;
        const Pixel *srow = src;
        for (int y = 0; y < h; y++, srow += ss, tmp += w)
            for (int x = 0; x < w; x += N) {
                const size_t n = (size_t) (w - x < N ? w - x : N);
                const auto v = BilinDot<kCap, kI16v>(d, srow + x, ss, n, v16,
                                                    vmy, vmy16, rnd, 4 - ib);
                StoreI16<kCap>(d, tmp + x, n, hn::Sub(v, vbias));
            }
    } else
        hwy_prep_c<Pixel, kCap>(tmp, src, ss, w, h, ib, prep_bias);
    };
    if (px_i16) run(std::true_type{});
    else run(std::false_type{});
}

// put_8tap_scaled_c from src/mc_tmpl.c. The C ring of 8 mid row pointers is
// replaced by absolute line indexing: the fill for counter in_y writes line
// in_y + 5 (relative to the unshifted src) into row in_y & 7, so the vertical
// pass for output row y (src_y = my >> 10) reads rows (src_y + k) & 7, and
// the unfiltered vertical path reads row (src_y + 3) & 7.
template <typename Pixel, int kCap>
static void hwy_put_8tap_scaled_impl(Pixel *const dst, const ptrdiff_t dst_stride,
                         const Pixel *const src, const ptrdiff_t src_stride,
                         const int w, const int h, const int mx, int my,
                         const int dx, const int dy, const int filter_type,
                         const int bitdepth_max,
                         const std::integral_constant<int, kCap>) {
    const int ib = sizeof(Pixel) == 1 ? 4 : 13 - hwy_ulog2(bitdepth_max);
    const ptrdiff_t ds = dst_stride / (ptrdiff_t) sizeof(Pixel);
    const ptrdiff_t ss = src_stride / (ptrdiff_t) sizeof(Pixel);
    using DT16 = hn::CappedTag<int16_t, kCap == 0 ? 8 : kCap>;
    using DT32 = hn::Repartition<int32_t, DT16>;
    const DT16 d;
    const int N = (int) hn::Lanes(d);
    int32_t hoff[16][8];
    hn::VFromD<DT16> htaps[16][8];
    const int nch = ScaledHInit(d, w, mx, dx, filter_type, hoff, htaps);

    int16_t mid[8 * 128];
    const Pixel *srow = src - 3 * ss;
    Pixel *drow = dst;
    int in_y = -8;
    for (int y = 0; y < h; y++) {
        const int src_y = my >> 10;
        const int vph = (my & 0x3ff) >> 6;
        const int8_t *const fv = !vph ? NULL :
            dav1d_mc_subpel_filters[h > 4 ? (filter_type >> 2)
                                          : (3 + ((filter_type >> 2) & 1))][vph - 1];
        while (in_y < src_y) {
            ScaledHRow8tap(d, srow, hoff, htaps, nch, mid + (in_y & 7) * 128,
                           (1 << (6 - ib)) >> 1, 6 - ib);
            srow += ss;
            in_y++;
        }
        if (fv) {
            hn::VFromD<DT16> vf[8]; LoadTaps(d, fv, vf);
            const int rnd_v = (1 << (6 + ib)) >> 1;
            for (int x = 0; x < w; x += N) {
                const size_t n = (size_t) (w - x < N ? w - x : N);
                hn::VFromD<DT16> rows[8];
                for (int k = 0; k < 8; k++)
                    rows[k] = LoadI16<kCap>(d, mid + ((src_y + k) & 7) * 128 + x, n);
                hn::VFromD<DT32> dot_e, dot_o;
                Dot8V<false>(d, rows, vf, dot_e, dot_o);
                StorePackPx<kCap>(d, drow + x, n, dot_e, dot_o, rnd_v, 6 + ib,
                                  bitdepth_max);
            }
        } else {
            const int16_t *const m3 = mid + ((src_y + 3) & 7) * 128;
            const auto vrnd = hn::Set(d, (1 << ib) >> 1);
            const auto vmax = hn::Set(d, bitdepth_max);
            for (int x = 0; x < w; x += N) {
                const size_t n = (size_t) (w - x < N ? w - x : N);
                const auto v = hn::ShiftRightSame(
                    hn::Add(LoadI16<kCap>(d, m3 + x, n), vrnd), ib);
                StorePx<kCap>(d, drow + x, n, hn::Clamp(v, hn::Zero(d), vmax));
            }
        }
        my += dy;
        drow += ds;
    }
}

// prep_8tap_scaled_c from src/mc_tmpl.c; same structure as the put variant.
template <typename Pixel, int kCap>
static void hwy_prep_8tap_scaled_impl(int16_t *tmp, const Pixel *const src,
                         const ptrdiff_t src_stride, const int w, const int h,
                         const int mx, int my, const int dx, const int dy,
                         const int filter_type, const int bitdepth_max,
                         const std::integral_constant<int, kCap>) {
    const int ib = sizeof(Pixel) == 1 ? 4 : 13 - hwy_ulog2(bitdepth_max);
    const int prep_bias = sizeof(Pixel) == 1 ? 0 : 8192;
    const ptrdiff_t ss = src_stride / (ptrdiff_t) sizeof(Pixel);
    using DT16 = hn::CappedTag<int16_t, kCap == 0 ? 8 : kCap>;
    using DT32 = hn::Repartition<int32_t, DT16>;
    const DT16 d;
    const int N = (int) hn::Lanes(d);
    int32_t hoff[16][8];
    hn::VFromD<DT16> htaps[16][8];
    const int nch = ScaledHInit(d, w, mx, dx, filter_type, hoff, htaps);

    int16_t mid[8 * 128];
    const Pixel *srow = src - 3 * ss;
    int16_t *trow = tmp;
    int in_y = -8;
    for (int y = 0; y < h; y++) {
        const int src_y = my >> 10;
        const int vph = (my & 0x3ff) >> 6;
        const int8_t *const fv = !vph ? NULL :
            dav1d_mc_subpel_filters[h > 4 ? (filter_type >> 2)
                                          : (3 + ((filter_type >> 2) & 1))][vph - 1];
        while (in_y < src_y) {
            ScaledHRow8tap(d, srow, hoff, htaps, nch, mid + (in_y & 7) * 128,
                           (1 << (6 - ib)) >> 1, 6 - ib);
            srow += ss;
            in_y++;
        }
        if (fv) {
            hn::VFromD<DT16> vf[8]; LoadTaps(d, fv, vf);
            for (int x = 0; x < w; x += N) {
                const size_t n = (size_t) (w - x < N ? w - x : N);
                hn::VFromD<DT16> rows[8];
                for (int k = 0; k < 8; k++)
                    rows[k] = LoadI16<kCap>(d, mid + ((src_y + k) & 7) * 128 + x, n);
                hn::VFromD<DT32> dot_e, dot_o;
                Dot8V<false>(d, rows, vf, dot_e, dot_o);
                // PREP_BIAS folded into the rounding constant (exact for
                // arithmetic shifts; result fits int16).
                StorePackI16<kCap>(d, trow + x, n, dot_e, dot_o,
                                   32 - (prep_bias << 6), 6);
            }
        } else {
            const int16_t *const m3 = mid + ((src_y + 3) & 7) * 128;
            const auto vbias = hn::Set(d, prep_bias);
            for (int x = 0; x < w; x += N) {
                const size_t n = (size_t) (w - x < N ? w - x : N);
                StoreI16<kCap>(d, trow + x, n,
                               hn::Sub(LoadI16<kCap>(d, m3 + x, n), vbias));
            }
        }
        my += dy;
        trow += w;
    }
}

// put_bilin_scaled_c from src/mc_tmpl.c; 2-row ring: the fill for counter
// in_y writes line in_y + 2 into row in_y & 1, so output row y uses rows
// src_y & 1 and (src_y + 1) & 1.
template <typename Pixel, int kCap>
static void hwy_put_bilin_scaled_impl(Pixel *const dst, const ptrdiff_t dst_stride,
                         const Pixel *const src, const ptrdiff_t src_stride,
                         const int w, const int h, const int mx, int my,
                         const int dx, const int dy, const int bitdepth_max,
                         const std::integral_constant<int, kCap>) {
    const int ib = sizeof(Pixel) == 1 ? 4 : 13 - hwy_ulog2(bitdepth_max);
    const ptrdiff_t ds = dst_stride / (ptrdiff_t) sizeof(Pixel);
    const ptrdiff_t ss = src_stride / (ptrdiff_t) sizeof(Pixel);
    using DT16 = hn::CappedTag<int16_t, kCap == 0 ? 8 : kCap>;
    using DT32 = hn::Repartition<int32_t, DT16>;
    const DT16 d;
    const int N = (int) hn::Lanes(d);
    const auto v16 = hn::Set(d, 16);
    int32_t hoff[16][8];
    hn::VFromD<DT16> hcm[16][2];
    const int nch = ScaledHInitBilin(d, w, mx, dx, hoff, hcm);

    int16_t mid[2 * 128];
    const Pixel *srow = src;
    Pixel *drow = dst;
    int in_y = -2;
    for (int y = 0; y < h; y++) {
        const int src_y = my >> 10;
        const auto vdmy = hn::Set(d, (my & 0x3ff) >> 6);
        while (in_y < src_y) {
            ScaledHRowBilin(d, srow, hoff, hcm, nch, mid + (in_y & 1) * 128,
                            (1 << (4 - ib)) >> 1, 4 - ib);
            srow += ss;
            in_y++;
        }
        const int16_t *const m1 = mid + (src_y & 1) * 128;
        const int16_t *const m2 = mid + ((src_y + 1) & 1) * 128;
        const int rnd_v = (1 << (4 + ib)) >> 1;
        for (int x = 0; x < w; x += N) {
            const size_t n = (size_t) (w - x < N ? w - x : N);
            hn::VFromD<DT32> dot_e, dot_o;
            DotBilinV(d, LoadI16<kCap>(d, m1 + x, n),
                      LoadI16<kCap>(d, m2 + x, n), v16, vdmy, dot_e, dot_o);
            StorePackPx<kCap>(d, drow + x, n, dot_e, dot_o, rnd_v, 4 + ib,
                              bitdepth_max);
        }
        my += dy;
        drow += ds;
    }
}

// prep_bilin_scaled_c from src/mc_tmpl.c.
template <typename Pixel, int kCap>
static void hwy_prep_bilin_scaled_impl(int16_t *tmp, const Pixel *const src,
                         const ptrdiff_t src_stride, const int w, const int h,
                         const int mx, int my, const int dx, const int dy,
                         const int bitdepth_max,
                         const std::integral_constant<int, kCap>) {
    const int ib = sizeof(Pixel) == 1 ? 4 : 13 - hwy_ulog2(bitdepth_max);
    const int prep_bias = sizeof(Pixel) == 1 ? 0 : 8192;
    const ptrdiff_t ss = src_stride / (ptrdiff_t) sizeof(Pixel);
    using DT16 = hn::CappedTag<int16_t, kCap == 0 ? 8 : kCap>;
    using DT32 = hn::Repartition<int32_t, DT16>;
    const DT16 d;
    const int N = (int) hn::Lanes(d);
    const auto v16 = hn::Set(d, 16);
    int32_t hoff[16][8];
    hn::VFromD<DT16> hcm[16][2];
    const int nch = ScaledHInitBilin(d, w, mx, dx, hoff, hcm);

    int16_t mid[2 * 128];
    const Pixel *srow = src;
    int16_t *trow = tmp;
    int in_y = -2;
    for (int y = 0; y < h; y++) {
        const int src_y = my >> 10;
        const auto vdmy = hn::Set(d, (my & 0x3ff) >> 6);
        while (in_y < src_y) {
            ScaledHRowBilin(d, srow, hoff, hcm, nch, mid + (in_y & 1) * 128,
                            (1 << (4 - ib)) >> 1, 4 - ib);
            srow += ss;
            in_y++;
        }
        const int16_t *const m1 = mid + (src_y & 1) * 128;
        const int16_t *const m2 = mid + ((src_y + 1) & 1) * 128;
        for (int x = 0; x < w; x += N) {
            const size_t n = (size_t) (w - x < N ? w - x : N);
            hn::VFromD<DT32> dot_e, dot_o;
            DotBilinV(d, LoadI16<kCap>(d, m1 + x, n),
                      LoadI16<kCap>(d, m2 + x, n), v16, vdmy, dot_e, dot_o);
            // PREP_BIAS folded into the rounding constant (exact).
            StorePackI16<kCap>(d, trow + x, n, dot_e, dot_o,
                               8 - (prep_bias << 4), 4);
        }
        my += dy;
        trow += w;
    }
}

// warp_affine_8x8_c / warp_affine_8x8t_c from src/mc_tmpl.c. Filter taps vary
// per column (tmx/tmy per 4x4 sub-block coordinate), so both passes use
// transposed per-column tap matrices (LoadTapsT). Intermediate ranges over dav1d_mc_warp_filter (max positive tap sum 175,
// negative -47): h-pass output in [-6015, 22395], v-pass dot in i32.
template <typename Pixel, bool kTmp, typename Dst>
static void hwy_warp8x8_impl(Dst *const dst, const ptrdiff_t dst_stride,
                             const Pixel *const src, const ptrdiff_t src_stride,
                             const int16_t *const abcd, int mx, int my,
                             const int bitdepth_max) {
    const int ib = sizeof(Pixel) == 1 ? 4 : 13 - hwy_ulog2(bitdepth_max);
    const ptrdiff_t ss = src_stride / (ptrdiff_t) sizeof(Pixel);
    const ptrdiff_t ds = kTmp ? dst_stride : dst_stride / (ptrdiff_t) sizeof(Pixel);
    const hn::CappedTag<int16_t, 8> d;

    int16_t mid[15 * 8];
    const Pixel *srow = src - 3 * ss;
    const int rnd_h = (1 << (7 - ib)) >> 1;
    const hn::Repartition<int32_t, decltype(d)> d32;
    for (int y = 0; y < 15; y++, mx += abcd[1], srow += ss) {
        const int8_t *frows[8];
        for (int x = 0, tmx = mx; x < 8; x++, tmx += abcd[0])
            frows[x] = dav1d_mc_warp_filter[64 + ((tmx + 512) >> 10)];
        hn::VFromD<decltype(d)> ft[8];
        LoadTapsT(d, frows, ft);
        // Contiguous window: lanes of v[k] are the per-column tap-k sources
        // (same two loads the C code reads, srow[-3 .. 11]).
        const auto v0 = LoadI16<8>(d, srow - 3, 8);
        const auto v8 = LoadI16<0>(d, srow + 5, 7);
        auto de0 = hn::Zero(d32), de1 = hn::Zero(d32);
        auto do0 = hn::Zero(d32), do1 = hn::Zero(d32);
        ForEachTap<0, 8>([&](auto kc) {
            constexpr int k = decltype(kc)::value;
            const auto v = ShiftedWindow<k>(d, v8, v0);
            auto& de = (k & 1) ? de1 : de0;
            auto& dout = (k & 1) ? do1 : do0;
            de = hn::Add(de, hn::MulEven(v, ft[k]));
            dout = hn::Add(dout, hn::MulOdd(v, ft[k]));
        });
        StoreI16<8>(d, mid + y * 8, 8,
                    RoundPack(d, hn::Add(de0, de1), hn::Add(do0, do1),
                              rnd_h, 7 - ib));
    }

    const int16_t *mrow = mid;
    Dst *drow = dst;
    for (int y = 0; y < 8; y++, my += abcd[3], mrow += 8, drow += ds) {
        const int8_t *frows[8];
        for (int x = 0, tmy = my; x < 8; x++, tmy += abcd[2])
            frows[x] = dav1d_mc_warp_filter[64 + ((tmy + 512) >> 10)];
        hn::VFromD<decltype(d)> ft[8];
        LoadTapsT(d, frows, ft);
        hn::VFromD<decltype(d)> rows[8];
        for (int k = 0; k < 8; k++)
            rows[k] = LoadI16<8>(d, mrow + k * 8, 8);
        auto de0 = hn::Zero(d32), de1 = hn::Zero(d32);
        auto do0 = hn::Zero(d32), do1 = hn::Zero(d32);
        for (int k = 0; k < 8; k++) {
            auto& de = (k & 1) ? de1 : de0;
            auto& dout = (k & 1) ? do1 : do0;
            de = hn::Add(de, hn::MulEven(rows[k], ft[k]));
            dout = hn::Add(dout, hn::MulOdd(rows[k], ft[k]));
        }
        const auto dot_e = hn::Add(de0, de1);
        const auto dot_o = hn::Add(do0, do1);
        if constexpr (kTmp) {
            // (dot + 64) >> 7 - PREP_BIAS, bias folded into the rounding
            // constant (exact for arithmetic shifts; result fits int16).
            constexpr int prep_bias = sizeof(Pixel) == 1 ? 0 : 8192;
            StorePackI16<8>(d, reinterpret_cast<int16_t *>(drow), 8,
                            dot_e, dot_o, 64 - (prep_bias << 7), 7);
        } else {
            StorePackPx<8>(d, reinterpret_cast<Pixel *>(drow), 8,
                           dot_e, dot_o, (1 << (7 + ib)) >> 1, 7 + ib,
                           bitdepth_max);
        }
    }
}

// Widths >= 8 are multiples of 8 in dav1d, so the partial-chunk path only
// ever sees w < 8.
template <typename Pixel>
static HWY_INLINE void hwy_put_8tap(Pixel *const dst, const ptrdiff_t dst_stride,
                                    const Pixel *const src, const ptrdiff_t src_stride,
                                    const int w, const int h, const int mx, const int my,
                                    const int filter_type, const int bitdepth_max) {
    if (w % 8 == 0)
        hwy_put_8tap_impl(dst, dst_stride, src, src_stride, w, h, mx, my,
                          filter_type, bitdepth_max, std::integral_constant<int, 8>{});
    else if (w % 4 == 0)
        hwy_put_8tap_impl(dst, dst_stride, src, src_stride, w, h, mx, my,
                          filter_type, bitdepth_max, std::integral_constant<int, 4>{});
    else
        hwy_put_8tap_impl(dst, dst_stride, src, src_stride, w, h, mx, my,
                          filter_type, bitdepth_max, std::integral_constant<int, 0>{});
}

template <typename Pixel>
static HWY_INLINE void hwy_prep_8tap(int16_t *const tmp, const Pixel *const src,
                                     const ptrdiff_t src_stride, const int w, const int h,
                                     const int mx, const int my, const int filter_type,
                                     const int bitdepth_max) {
    if (w % 8 == 0)
        hwy_prep_8tap_impl(tmp, src, src_stride, w, h, mx, my,
                           filter_type, bitdepth_max, std::integral_constant<int, 8>{});
    else if (w % 4 == 0)
        hwy_prep_8tap_impl(tmp, src, src_stride, w, h, mx, my,
                           filter_type, bitdepth_max, std::integral_constant<int, 4>{});
    else
        hwy_prep_8tap_impl(tmp, src, src_stride, w, h, mx, my,
                           filter_type, bitdepth_max, std::integral_constant<int, 0>{});
}

template <typename Pixel>
static HWY_INLINE void hwy_put_bilin(Pixel *const dst, const ptrdiff_t dst_stride,
                                     const Pixel *const src, const ptrdiff_t src_stride,
                                     const int w, const int h, const int mx, const int my,
                                     const int bitdepth_max) {
    if (w % 8 == 0)
        hwy_put_bilin_impl(dst, dst_stride, src, src_stride, w, h, mx, my,
                           bitdepth_max, std::integral_constant<int, 8>{});
    else if (w % 4 == 0)
        hwy_put_bilin_impl(dst, dst_stride, src, src_stride, w, h, mx, my,
                           bitdepth_max, std::integral_constant<int, 4>{});
    else
        hwy_put_bilin_impl(dst, dst_stride, src, src_stride, w, h, mx, my,
                           bitdepth_max, std::integral_constant<int, 0>{});
}

template <typename Pixel>
static HWY_INLINE void hwy_prep_bilin(int16_t *const tmp, const Pixel *const src,
                                      const ptrdiff_t src_stride, const int w, const int h,
                                      const int mx, const int my, const int bitdepth_max) {
    if (w % 8 == 0)
        hwy_prep_bilin_impl(tmp, src, src_stride, w, h, mx, my, bitdepth_max,
                            std::integral_constant<int, 8>{});
    else if (w % 4 == 0)
        hwy_prep_bilin_impl(tmp, src, src_stride, w, h, mx, my, bitdepth_max,
                            std::integral_constant<int, 4>{});
    else
        hwy_prep_bilin_impl(tmp, src, src_stride, w, h, mx, my, bitdepth_max,
                            std::integral_constant<int, 0>{});
}

// The scaled kernels filter one 8-column chunk at a time; the kCap == 0
// (partial) path only ever sees w < 8.
template <typename Pixel>
static HWY_INLINE void hwy_put_8tap_scaled(Pixel *const dst, const ptrdiff_t dst_stride,
                                    const Pixel *const src, const ptrdiff_t src_stride,
                                    const int w, const int h, const int mx, const int my,
                                    const int dx, const int dy, const int filter_type,
                                    const int bitdepth_max) {
    if (w % 8 == 0)
        hwy_put_8tap_scaled_impl(dst, dst_stride, src, src_stride, w, h, mx, my,
                                 dx, dy, filter_type, bitdepth_max,
                                 std::integral_constant<int, 8>{});
    else
        hwy_put_8tap_scaled_impl(dst, dst_stride, src, src_stride, w, h, mx, my,
                                 dx, dy, filter_type, bitdepth_max,
                                 std::integral_constant<int, 0>{});
}

template <typename Pixel>
static HWY_INLINE void hwy_prep_8tap_scaled(int16_t *const tmp, const Pixel *const src,
                                     const ptrdiff_t src_stride, const int w, const int h,
                                     const int mx, const int my, const int dx,
                                     const int dy, const int filter_type,
                                     const int bitdepth_max) {
    if (w % 8 == 0)
        hwy_prep_8tap_scaled_impl(tmp, src, src_stride, w, h, mx, my, dx, dy,
                                  filter_type, bitdepth_max,
                                  std::integral_constant<int, 8>{});
    else
        hwy_prep_8tap_scaled_impl(tmp, src, src_stride, w, h, mx, my, dx, dy,
                                  filter_type, bitdepth_max,
                                  std::integral_constant<int, 0>{});
}

template <typename Pixel>
static HWY_INLINE void hwy_put_bilin_scaled(Pixel *const dst, const ptrdiff_t dst_stride,
                                     const Pixel *const src, const ptrdiff_t src_stride,
                                     const int w, const int h, const int mx, const int my,
                                     const int dx, const int dy,
                                     const int bitdepth_max) {
    if (w % 8 == 0)
        hwy_put_bilin_scaled_impl(dst, dst_stride, src, src_stride, w, h, mx, my,
                                  dx, dy, bitdepth_max,
                                  std::integral_constant<int, 8>{});
    else
        hwy_put_bilin_scaled_impl(dst, dst_stride, src, src_stride, w, h, mx, my,
                                  dx, dy, bitdepth_max,
                                  std::integral_constant<int, 0>{});
}

template <typename Pixel>
static HWY_INLINE void hwy_prep_bilin_scaled(int16_t *const tmp, const Pixel *const src,
                                      const ptrdiff_t src_stride, const int w, const int h,
                                      const int mx, const int my, const int dx,
                                      const int dy, const int bitdepth_max) {
    if (w % 8 == 0)
        hwy_prep_bilin_scaled_impl(tmp, src, src_stride, w, h, mx, my, dx, dy,
                                   bitdepth_max, std::integral_constant<int, 8>{});
    else
        hwy_prep_bilin_scaled_impl(tmp, src, src_stride, w, h, mx, my, dx, dy,
                                   bitdepth_max, std::integral_constant<int, 0>{});
}

#define HWY_MC_FILTER_FNS(bpc, sfx, HIGHBD_SUFFIX, BD_MAX) \
void put_8tap_regular_##sfx(uint##bpc##_t *dst, const ptrdiff_t dst_stride, \
                            const uint##bpc##_t *src, const ptrdiff_t src_stride, \
                            const int w, const int h, const int mx, const int my \
                            HIGHBD_SUFFIX) { \
    hwy_put_8tap(dst, dst_stride, src, src_stride, w, h, mx, my, \
                 kFilterRegular | (kFilterRegular << 2), BD_MAX); \
} \
void put_8tap_regular_smooth_##sfx(uint##bpc##_t *dst, const ptrdiff_t dst_stride, \
                            const uint##bpc##_t *src, const ptrdiff_t src_stride, \
                            const int w, const int h, const int mx, const int my \
                            HIGHBD_SUFFIX) { \
    hwy_put_8tap(dst, dst_stride, src, src_stride, w, h, mx, my, \
                 kFilterRegular | (kFilterSmooth << 2), BD_MAX); \
} \
void put_8tap_regular_sharp_##sfx(uint##bpc##_t *dst, const ptrdiff_t dst_stride, \
                            const uint##bpc##_t *src, const ptrdiff_t src_stride, \
                            const int w, const int h, const int mx, const int my \
                            HIGHBD_SUFFIX) { \
    hwy_put_8tap(dst, dst_stride, src, src_stride, w, h, mx, my, \
                 kFilterRegular | (kFilterSharp << 2), BD_MAX); \
} \
void put_8tap_sharp_regular_##sfx(uint##bpc##_t *dst, const ptrdiff_t dst_stride, \
                            const uint##bpc##_t *src, const ptrdiff_t src_stride, \
                            const int w, const int h, const int mx, const int my \
                            HIGHBD_SUFFIX) { \
    hwy_put_8tap(dst, dst_stride, src, src_stride, w, h, mx, my, \
                 kFilterSharp | (kFilterRegular << 2), BD_MAX); \
} \
void put_8tap_sharp_smooth_##sfx(uint##bpc##_t *dst, const ptrdiff_t dst_stride, \
                            const uint##bpc##_t *src, const ptrdiff_t src_stride, \
                            const int w, const int h, const int mx, const int my \
                            HIGHBD_SUFFIX) { \
    hwy_put_8tap(dst, dst_stride, src, src_stride, w, h, mx, my, \
                 kFilterSharp | (kFilterSmooth << 2), BD_MAX); \
} \
void put_8tap_sharp_##sfx(uint##bpc##_t *dst, const ptrdiff_t dst_stride, \
                            const uint##bpc##_t *src, const ptrdiff_t src_stride, \
                            const int w, const int h, const int mx, const int my \
                            HIGHBD_SUFFIX) { \
    hwy_put_8tap(dst, dst_stride, src, src_stride, w, h, mx, my, \
                 kFilterSharp | (kFilterSharp << 2), BD_MAX); \
} \
void put_8tap_smooth_regular_##sfx(uint##bpc##_t *dst, const ptrdiff_t dst_stride, \
                            const uint##bpc##_t *src, const ptrdiff_t src_stride, \
                            const int w, const int h, const int mx, const int my \
                            HIGHBD_SUFFIX) { \
    hwy_put_8tap(dst, dst_stride, src, src_stride, w, h, mx, my, \
                 kFilterSmooth | (kFilterRegular << 2), BD_MAX); \
} \
void put_8tap_smooth_##sfx(uint##bpc##_t *dst, const ptrdiff_t dst_stride, \
                            const uint##bpc##_t *src, const ptrdiff_t src_stride, \
                            const int w, const int h, const int mx, const int my \
                            HIGHBD_SUFFIX) { \
    hwy_put_8tap(dst, dst_stride, src, src_stride, w, h, mx, my, \
                 kFilterSmooth | (kFilterSmooth << 2), BD_MAX); \
} \
void put_8tap_smooth_sharp_##sfx(uint##bpc##_t *dst, const ptrdiff_t dst_stride, \
                            const uint##bpc##_t *src, const ptrdiff_t src_stride, \
                            const int w, const int h, const int mx, const int my \
                            HIGHBD_SUFFIX) { \
    hwy_put_8tap(dst, dst_stride, src, src_stride, w, h, mx, my, \
                 kFilterSmooth | (kFilterSharp << 2), BD_MAX); \
} \
void put_bilin_##sfx(uint##bpc##_t *dst, const ptrdiff_t dst_stride, \
                     const uint##bpc##_t *src, const ptrdiff_t src_stride, \
                     const int w, const int h, const int mx, const int my \
                     HIGHBD_SUFFIX) { \
    hwy_put_bilin(dst, dst_stride, src, src_stride, w, h, mx, my, BD_MAX); \
} \
void prep_8tap_regular_##sfx(int16_t *tmp, const uint##bpc##_t *src, \
                             const ptrdiff_t src_stride, const int w, const int h, \
                             const int mx, const int my HIGHBD_SUFFIX) { \
    hwy_prep_8tap(tmp, src, src_stride, w, h, mx, my, \
                  kFilterRegular | (kFilterRegular << 2), BD_MAX); \
} \
void prep_8tap_regular_smooth_##sfx(int16_t *tmp, const uint##bpc##_t *src, \
                             const ptrdiff_t src_stride, const int w, const int h, \
                             const int mx, const int my HIGHBD_SUFFIX) { \
    hwy_prep_8tap(tmp, src, src_stride, w, h, mx, my, \
                  kFilterRegular | (kFilterSmooth << 2), BD_MAX); \
} \
void prep_8tap_regular_sharp_##sfx(int16_t *tmp, const uint##bpc##_t *src, \
                             const ptrdiff_t src_stride, const int w, const int h, \
                             const int mx, const int my HIGHBD_SUFFIX) { \
    hwy_prep_8tap(tmp, src, src_stride, w, h, mx, my, \
                  kFilterRegular | (kFilterSharp << 2), BD_MAX); \
} \
void prep_8tap_sharp_regular_##sfx(int16_t *tmp, const uint##bpc##_t *src, \
                             const ptrdiff_t src_stride, const int w, const int h, \
                             const int mx, const int my HIGHBD_SUFFIX) { \
    hwy_prep_8tap(tmp, src, src_stride, w, h, mx, my, \
                  kFilterSharp | (kFilterRegular << 2), BD_MAX); \
} \
void prep_8tap_sharp_smooth_##sfx(int16_t *tmp, const uint##bpc##_t *src, \
                             const ptrdiff_t src_stride, const int w, const int h, \
                             const int mx, const int my HIGHBD_SUFFIX) { \
    hwy_prep_8tap(tmp, src, src_stride, w, h, mx, my, \
                  kFilterSharp | (kFilterSmooth << 2), BD_MAX); \
} \
void prep_8tap_sharp_##sfx(int16_t *tmp, const uint##bpc##_t *src, \
                             const ptrdiff_t src_stride, const int w, const int h, \
                             const int mx, const int my HIGHBD_SUFFIX) { \
    hwy_prep_8tap(tmp, src, src_stride, w, h, mx, my, \
                  kFilterSharp | (kFilterSharp << 2), BD_MAX); \
} \
void prep_8tap_smooth_regular_##sfx(int16_t *tmp, const uint##bpc##_t *src, \
                             const ptrdiff_t src_stride, const int w, const int h, \
                             const int mx, const int my HIGHBD_SUFFIX) { \
    hwy_prep_8tap(tmp, src, src_stride, w, h, mx, my, \
                  kFilterSmooth | (kFilterRegular << 2), BD_MAX); \
} \
void prep_8tap_smooth_##sfx(int16_t *tmp, const uint##bpc##_t *src, \
                             const ptrdiff_t src_stride, const int w, const int h, \
                             const int mx, const int my HIGHBD_SUFFIX) { \
    hwy_prep_8tap(tmp, src, src_stride, w, h, mx, my, \
                  kFilterSmooth | (kFilterSmooth << 2), BD_MAX); \
} \
void prep_8tap_smooth_sharp_##sfx(int16_t *tmp, const uint##bpc##_t *src, \
                             const ptrdiff_t src_stride, const int w, const int h, \
                             const int mx, const int my HIGHBD_SUFFIX) { \
    hwy_prep_8tap(tmp, src, src_stride, w, h, mx, my, \
                  kFilterSmooth | (kFilterSharp << 2), BD_MAX); \
} \
void prep_bilin_##sfx(int16_t *tmp, const uint##bpc##_t *src, \
                      const ptrdiff_t src_stride, const int w, const int h, \
                      const int mx, const int my HIGHBD_SUFFIX) { \
    hwy_prep_bilin(tmp, src, src_stride, w, h, mx, my, BD_MAX); \
}

/* HIGHBD_SUFFIX/BD_MAX are referenced as ambient macros, not passed along,
 * so that their expansions are not pre-expanded into macro arguments. */
#define HWY_MC_SCALED_PUT_FN(name, sfx, FT, bpc) \
void put_8tap_##name##_scaled_##sfx(uint##bpc##_t *dst, const ptrdiff_t dst_stride, \
                            const uint##bpc##_t *src, const ptrdiff_t src_stride, \
                            const int w, const int h, const int mx, const int my, \
                            const int dx, const int dy HIGHBD_SUFFIX) { \
    hwy_put_8tap_scaled(dst, dst_stride, src, src_stride, w, h, mx, my, dx, dy, \
                        FT, BD_MAX); \
} \
void prep_8tap_##name##_scaled_##sfx(int16_t *tmp, const uint##bpc##_t *src, \
                             const ptrdiff_t src_stride, const int w, const int h, \
                             const int mx, const int my, const int dx, \
                             const int dy HIGHBD_SUFFIX) { \
    hwy_prep_8tap_scaled(tmp, src, src_stride, w, h, mx, my, dx, dy, \
                         FT, BD_MAX); \
}

#define HWY_MC_SCALED_FNS(bpc, sfx) \
HWY_MC_SCALED_PUT_FN(regular,        sfx, kFilterRegular | (kFilterRegular << 2), bpc) \
HWY_MC_SCALED_PUT_FN(regular_smooth, sfx, kFilterRegular | (kFilterSmooth << 2),  bpc) \
HWY_MC_SCALED_PUT_FN(regular_sharp,  sfx, kFilterRegular | (kFilterSharp << 2),   bpc) \
HWY_MC_SCALED_PUT_FN(sharp_regular,  sfx, kFilterSharp | (kFilterRegular << 2),   bpc) \
HWY_MC_SCALED_PUT_FN(sharp_smooth,   sfx, kFilterSharp | (kFilterSmooth << 2),    bpc) \
HWY_MC_SCALED_PUT_FN(sharp,          sfx, kFilterSharp | (kFilterSharp << 2),     bpc) \
HWY_MC_SCALED_PUT_FN(smooth_regular, sfx, kFilterSmooth | (kFilterRegular << 2),  bpc) \
HWY_MC_SCALED_PUT_FN(smooth,         sfx, kFilterSmooth | (kFilterSmooth << 2),   bpc) \
HWY_MC_SCALED_PUT_FN(smooth_sharp,   sfx, kFilterSmooth | (kFilterSharp << 2),    bpc) \
void put_bilin_scaled_##sfx(uint##bpc##_t *dst, const ptrdiff_t dst_stride, \
                     const uint##bpc##_t *src, const ptrdiff_t src_stride, \
                     const int w, const int h, const int mx, const int my, \
                     const int dx, const int dy HIGHBD_SUFFIX) { \
    hwy_put_bilin_scaled(dst, dst_stride, src, src_stride, w, h, mx, my, \
                         dx, dy, BD_MAX); \
} \
void prep_bilin_scaled_##sfx(int16_t *tmp, const uint##bpc##_t *src, \
                      const ptrdiff_t src_stride, const int w, const int h, \
                      const int mx, const int my, const int dx, const int dy \
                      HIGHBD_SUFFIX) { \
    hwy_prep_bilin_scaled(tmp, src, src_stride, w, h, mx, my, dx, dy, BD_MAX); \
} \
void warp_affine_8x8_##sfx(uint##bpc##_t *dst, const ptrdiff_t dst_stride, \
                    const uint##bpc##_t *src, const ptrdiff_t src_stride, \
                    const int16_t *const abcd, const int mx, const int my \
                    HIGHBD_SUFFIX) { \
    hwy_warp8x8_impl<uint##bpc##_t, false>(dst, dst_stride, src, src_stride, \
                                           abcd, mx, my, BD_MAX); \
} \
void warp_affine_8x8t_##sfx(int16_t *tmp, const ptrdiff_t tmp_stride, \
                     const uint##bpc##_t *src, const ptrdiff_t src_stride, \
                     const int16_t *const abcd, const int mx, const int my \
                     HIGHBD_SUFFIX) { \
    hwy_warp8x8_impl<uint##bpc##_t, true>(tmp, tmp_stride, src, src_stride, \
                                          abcd, mx, my, BD_MAX); \
}

#define HIGHBD_SUFFIX
#define BD_MAX 255
HWY_MC_FILTER_FNS(8, 8bpc, HIGHBD_SUFFIX, BD_MAX)
HWY_MC_SCALED_FNS(8, 8bpc)
#undef HIGHBD_SUFFIX
#undef BD_MAX
#define HIGHBD_SUFFIX , const int bitdepth_max
#define BD_MAX bitdepth_max
HWY_MC_FILTER_FNS(16, 16bpc, HIGHBD_SUFFIX, BD_MAX)
HWY_MC_SCALED_FNS(16, 16bpc)
#undef HIGHBD_SUFFIX
#undef BD_MAX
#undef HWY_MC_FILTER_FNS
#undef HWY_MC_SCALED_FNS
#undef HWY_MC_SCALED_PUT_FN

}  // namespace HWY_NAMESPACE
}  // namespace dav1d

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace dav1d {
HWY_EXPORT(put_8tap_regular_8bpc);
HWY_EXPORT(put_8tap_regular_smooth_8bpc);
HWY_EXPORT(put_8tap_regular_sharp_8bpc);
HWY_EXPORT(put_8tap_sharp_regular_8bpc);
HWY_EXPORT(put_8tap_sharp_smooth_8bpc);
HWY_EXPORT(put_8tap_sharp_8bpc);
HWY_EXPORT(put_8tap_smooth_regular_8bpc);
HWY_EXPORT(put_8tap_smooth_8bpc);
HWY_EXPORT(put_8tap_smooth_sharp_8bpc);
HWY_EXPORT(put_bilin_8bpc);
HWY_EXPORT(prep_8tap_regular_8bpc);
HWY_EXPORT(prep_8tap_regular_smooth_8bpc);
HWY_EXPORT(prep_8tap_regular_sharp_8bpc);
HWY_EXPORT(prep_8tap_sharp_regular_8bpc);
HWY_EXPORT(prep_8tap_sharp_smooth_8bpc);
HWY_EXPORT(prep_8tap_sharp_8bpc);
HWY_EXPORT(prep_8tap_smooth_regular_8bpc);
HWY_EXPORT(prep_8tap_smooth_8bpc);
HWY_EXPORT(prep_8tap_smooth_sharp_8bpc);
HWY_EXPORT(prep_bilin_8bpc);
HWY_EXPORT(put_8tap_regular_scaled_8bpc);
HWY_EXPORT(put_8tap_regular_smooth_scaled_8bpc);
HWY_EXPORT(put_8tap_regular_sharp_scaled_8bpc);
HWY_EXPORT(put_8tap_sharp_regular_scaled_8bpc);
HWY_EXPORT(put_8tap_sharp_smooth_scaled_8bpc);
HWY_EXPORT(put_8tap_sharp_scaled_8bpc);
HWY_EXPORT(put_8tap_smooth_regular_scaled_8bpc);
HWY_EXPORT(put_8tap_smooth_scaled_8bpc);
HWY_EXPORT(put_8tap_smooth_sharp_scaled_8bpc);
HWY_EXPORT(put_bilin_scaled_8bpc);
HWY_EXPORT(prep_8tap_regular_scaled_8bpc);
HWY_EXPORT(prep_8tap_regular_smooth_scaled_8bpc);
HWY_EXPORT(prep_8tap_regular_sharp_scaled_8bpc);
HWY_EXPORT(prep_8tap_sharp_regular_scaled_8bpc);
HWY_EXPORT(prep_8tap_sharp_smooth_scaled_8bpc);
HWY_EXPORT(prep_8tap_sharp_scaled_8bpc);
HWY_EXPORT(prep_8tap_smooth_regular_scaled_8bpc);
HWY_EXPORT(prep_8tap_smooth_scaled_8bpc);
HWY_EXPORT(prep_8tap_smooth_sharp_scaled_8bpc);
HWY_EXPORT(prep_bilin_scaled_8bpc);
HWY_EXPORT(warp_affine_8x8_8bpc);
HWY_EXPORT(warp_affine_8x8t_8bpc);
HWY_EXPORT(put_8tap_regular_16bpc);
HWY_EXPORT(put_8tap_regular_smooth_16bpc);
HWY_EXPORT(put_8tap_regular_sharp_16bpc);
HWY_EXPORT(put_8tap_sharp_regular_16bpc);
HWY_EXPORT(put_8tap_sharp_smooth_16bpc);
HWY_EXPORT(put_8tap_sharp_16bpc);
HWY_EXPORT(put_8tap_smooth_regular_16bpc);
HWY_EXPORT(put_8tap_smooth_16bpc);
HWY_EXPORT(put_8tap_smooth_sharp_16bpc);
HWY_EXPORT(put_bilin_16bpc);
HWY_EXPORT(prep_8tap_regular_16bpc);
HWY_EXPORT(prep_8tap_regular_smooth_16bpc);
HWY_EXPORT(prep_8tap_regular_sharp_16bpc);
HWY_EXPORT(prep_8tap_sharp_regular_16bpc);
HWY_EXPORT(prep_8tap_sharp_smooth_16bpc);
HWY_EXPORT(prep_8tap_sharp_16bpc);
HWY_EXPORT(prep_8tap_smooth_regular_16bpc);
HWY_EXPORT(prep_8tap_smooth_16bpc);
HWY_EXPORT(prep_8tap_smooth_sharp_16bpc);
HWY_EXPORT(prep_bilin_16bpc);
HWY_EXPORT(put_8tap_regular_scaled_16bpc);
HWY_EXPORT(put_8tap_regular_smooth_scaled_16bpc);
HWY_EXPORT(put_8tap_regular_sharp_scaled_16bpc);
HWY_EXPORT(put_8tap_sharp_regular_scaled_16bpc);
HWY_EXPORT(put_8tap_sharp_smooth_scaled_16bpc);
HWY_EXPORT(put_8tap_sharp_scaled_16bpc);
HWY_EXPORT(put_8tap_smooth_regular_scaled_16bpc);
HWY_EXPORT(put_8tap_smooth_scaled_16bpc);
HWY_EXPORT(put_8tap_smooth_sharp_scaled_16bpc);
HWY_EXPORT(put_bilin_scaled_16bpc);
HWY_EXPORT(prep_8tap_regular_scaled_16bpc);
HWY_EXPORT(prep_8tap_regular_smooth_scaled_16bpc);
HWY_EXPORT(prep_8tap_regular_sharp_scaled_16bpc);
HWY_EXPORT(prep_8tap_sharp_regular_scaled_16bpc);
HWY_EXPORT(prep_8tap_sharp_smooth_scaled_16bpc);
HWY_EXPORT(prep_8tap_sharp_scaled_16bpc);
HWY_EXPORT(prep_8tap_smooth_regular_scaled_16bpc);
HWY_EXPORT(prep_8tap_smooth_scaled_16bpc);
HWY_EXPORT(prep_8tap_smooth_sharp_scaled_16bpc);
HWY_EXPORT(prep_bilin_scaled_16bpc);
HWY_EXPORT(warp_affine_8x8_16bpc);
HWY_EXPORT(warp_affine_8x8t_16bpc);
}  // namespace dav1d

namespace {
// Prefix of Dav1dMCDSPContext (src/mc.h), so that this file does not need
// dav1d's bitdepth-templated C headers; the compound/avg/mask/blend slots in
// between are only skipped over, never read or written here.
using McFn8 = void (*)(uint8_t *, ptrdiff_t, const uint8_t *, ptrdiff_t,
                       int, int, int, int);
using McScaledFn8 = void (*)(uint8_t *, ptrdiff_t, const uint8_t *, ptrdiff_t,
                             int, int, int, int, int, int);
using MctFn8 = void (*)(int16_t *, const uint8_t *, ptrdiff_t,
                        int, int, int, int);
using MctScaledFn8 = void (*)(int16_t *, const uint8_t *, ptrdiff_t,
                              int, int, int, int, int, int);
using AvgFn8 = void (*)(uint8_t *, ptrdiff_t, const int16_t *, const int16_t *,
                        int, int);
using WAvgFn8 = void (*)(uint8_t *, ptrdiff_t, const int16_t *,
                         const int16_t *, int, int, int);
using MaskFn8 = void (*)(uint8_t *, ptrdiff_t, const int16_t *,
                         const int16_t *, int, int, const uint8_t *);
using WMaskFn8 = void (*)(uint8_t *, ptrdiff_t, const int16_t *,
                          const int16_t *, int, int, uint8_t *, int);
using BlendFn8 = void (*)(uint8_t *, ptrdiff_t, const uint8_t *, int, int,
                          const uint8_t *);
using BlendDirFn8 = void (*)(uint8_t *, ptrdiff_t, const uint8_t *, int, int);
using WarpFn8 = void (*)(uint8_t *, ptrdiff_t, const uint8_t *, ptrdiff_t,
                         const int16_t *, int, int);
using WarpTFn8 = void (*)(int16_t *, ptrdiff_t, const uint8_t *, ptrdiff_t,
                          const int16_t *, int, int);
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
    WarpFn8 warp8x8;
    WarpTFn8 warp8x8t;
};

using McFn16 = void (*)(uint16_t *, ptrdiff_t, const uint16_t *, ptrdiff_t,
                        int, int, int, int, int);
using McScaledFn16 = void (*)(uint16_t *, ptrdiff_t, const uint16_t *, ptrdiff_t,
                              int, int, int, int, int, int, int);
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
using BlendDirFn16 = void (*)(uint16_t *, ptrdiff_t, const uint16_t *, int, int);
using WarpFn16 = void (*)(uint16_t *, ptrdiff_t, const uint16_t *, ptrdiff_t,
                          const int16_t *, int, int, int);
using WarpTFn16 = void (*)(int16_t *, ptrdiff_t, const uint16_t *, ptrdiff_t,
                           const int16_t *, int, int, int);
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
    WarpFn16 warp8x8;
    WarpTFn16 warp8x8t;
};
}  // namespace

namespace dav1d {

// enum Filter2d order in src/levels.h (horizontal-major).
static void mc_dsp_init_8bpc_hwy(void *const c) {
    auto *const ctx = static_cast<McDSP8 *>(c);
    ctx->mc[0] = HWY_DYNAMIC_POINTER(put_8tap_regular_8bpc);
    ctx->mc[1] = HWY_DYNAMIC_POINTER(put_8tap_regular_smooth_8bpc);
    ctx->mc[2] = HWY_DYNAMIC_POINTER(put_8tap_regular_sharp_8bpc);
    ctx->mc[3] = HWY_DYNAMIC_POINTER(put_8tap_sharp_regular_8bpc);
    ctx->mc[4] = HWY_DYNAMIC_POINTER(put_8tap_sharp_smooth_8bpc);
    ctx->mc[5] = HWY_DYNAMIC_POINTER(put_8tap_sharp_8bpc);
    ctx->mc[6] = HWY_DYNAMIC_POINTER(put_8tap_smooth_regular_8bpc);
    ctx->mc[7] = HWY_DYNAMIC_POINTER(put_8tap_smooth_8bpc);
    ctx->mc[8] = HWY_DYNAMIC_POINTER(put_8tap_smooth_sharp_8bpc);
    ctx->mc[9] = HWY_DYNAMIC_POINTER(put_bilin_8bpc);
    ctx->mct[0] = HWY_DYNAMIC_POINTER(prep_8tap_regular_8bpc);
    ctx->mct[1] = HWY_DYNAMIC_POINTER(prep_8tap_regular_smooth_8bpc);
    ctx->mct[2] = HWY_DYNAMIC_POINTER(prep_8tap_regular_sharp_8bpc);
    ctx->mct[3] = HWY_DYNAMIC_POINTER(prep_8tap_sharp_regular_8bpc);
    ctx->mct[4] = HWY_DYNAMIC_POINTER(prep_8tap_sharp_smooth_8bpc);
    ctx->mct[5] = HWY_DYNAMIC_POINTER(prep_8tap_sharp_8bpc);
    ctx->mct[6] = HWY_DYNAMIC_POINTER(prep_8tap_smooth_regular_8bpc);
    ctx->mct[7] = HWY_DYNAMIC_POINTER(prep_8tap_smooth_8bpc);
    ctx->mct[8] = HWY_DYNAMIC_POINTER(prep_8tap_smooth_sharp_8bpc);
    ctx->mct[9] = HWY_DYNAMIC_POINTER(prep_bilin_8bpc);
    ctx->mc_scaled[0] = HWY_DYNAMIC_POINTER(put_8tap_regular_scaled_8bpc);
    ctx->mc_scaled[1] = HWY_DYNAMIC_POINTER(put_8tap_regular_smooth_scaled_8bpc);
    ctx->mc_scaled[2] = HWY_DYNAMIC_POINTER(put_8tap_regular_sharp_scaled_8bpc);
    ctx->mc_scaled[3] = HWY_DYNAMIC_POINTER(put_8tap_sharp_regular_scaled_8bpc);
    ctx->mc_scaled[4] = HWY_DYNAMIC_POINTER(put_8tap_sharp_smooth_scaled_8bpc);
    ctx->mc_scaled[5] = HWY_DYNAMIC_POINTER(put_8tap_sharp_scaled_8bpc);
    ctx->mc_scaled[6] = HWY_DYNAMIC_POINTER(put_8tap_smooth_regular_scaled_8bpc);
    ctx->mc_scaled[7] = HWY_DYNAMIC_POINTER(put_8tap_smooth_scaled_8bpc);
    ctx->mc_scaled[8] = HWY_DYNAMIC_POINTER(put_8tap_smooth_sharp_scaled_8bpc);
    ctx->mc_scaled[9] = HWY_DYNAMIC_POINTER(put_bilin_scaled_8bpc);
    ctx->mct_scaled[0] = HWY_DYNAMIC_POINTER(prep_8tap_regular_scaled_8bpc);
    ctx->mct_scaled[1] = HWY_DYNAMIC_POINTER(prep_8tap_regular_smooth_scaled_8bpc);
    ctx->mct_scaled[2] = HWY_DYNAMIC_POINTER(prep_8tap_regular_sharp_scaled_8bpc);
    ctx->mct_scaled[3] = HWY_DYNAMIC_POINTER(prep_8tap_sharp_regular_scaled_8bpc);
    ctx->mct_scaled[4] = HWY_DYNAMIC_POINTER(prep_8tap_sharp_smooth_scaled_8bpc);
    ctx->mct_scaled[5] = HWY_DYNAMIC_POINTER(prep_8tap_sharp_scaled_8bpc);
    ctx->mct_scaled[6] = HWY_DYNAMIC_POINTER(prep_8tap_smooth_regular_scaled_8bpc);
    ctx->mct_scaled[7] = HWY_DYNAMIC_POINTER(prep_8tap_smooth_scaled_8bpc);
    ctx->mct_scaled[8] = HWY_DYNAMIC_POINTER(prep_8tap_smooth_sharp_scaled_8bpc);
    ctx->mct_scaled[9] = HWY_DYNAMIC_POINTER(prep_bilin_scaled_8bpc);
    ctx->warp8x8 = HWY_DYNAMIC_POINTER(warp_affine_8x8_8bpc);
    ctx->warp8x8t = HWY_DYNAMIC_POINTER(warp_affine_8x8t_8bpc);
}

static void mc_dsp_init_16bpc_hwy(void *const c) {
    auto *const ctx = static_cast<McDSP16 *>(c);
    ctx->mc[0] = HWY_DYNAMIC_POINTER(put_8tap_regular_16bpc);
    ctx->mc[1] = HWY_DYNAMIC_POINTER(put_8tap_regular_smooth_16bpc);
    ctx->mc[2] = HWY_DYNAMIC_POINTER(put_8tap_regular_sharp_16bpc);
    ctx->mc[3] = HWY_DYNAMIC_POINTER(put_8tap_sharp_regular_16bpc);
    ctx->mc[4] = HWY_DYNAMIC_POINTER(put_8tap_sharp_smooth_16bpc);
    ctx->mc[5] = HWY_DYNAMIC_POINTER(put_8tap_sharp_16bpc);
    ctx->mc[6] = HWY_DYNAMIC_POINTER(put_8tap_smooth_regular_16bpc);
    ctx->mc[7] = HWY_DYNAMIC_POINTER(put_8tap_smooth_16bpc);
    ctx->mc[8] = HWY_DYNAMIC_POINTER(put_8tap_smooth_sharp_16bpc);
    ctx->mc[9] = HWY_DYNAMIC_POINTER(put_bilin_16bpc);
    ctx->mct[0] = HWY_DYNAMIC_POINTER(prep_8tap_regular_16bpc);
    ctx->mct[1] = HWY_DYNAMIC_POINTER(prep_8tap_regular_smooth_16bpc);
    ctx->mct[2] = HWY_DYNAMIC_POINTER(prep_8tap_regular_sharp_16bpc);
    ctx->mct[3] = HWY_DYNAMIC_POINTER(prep_8tap_sharp_regular_16bpc);
    ctx->mct[4] = HWY_DYNAMIC_POINTER(prep_8tap_sharp_smooth_16bpc);
    ctx->mct[5] = HWY_DYNAMIC_POINTER(prep_8tap_sharp_16bpc);
    ctx->mct[6] = HWY_DYNAMIC_POINTER(prep_8tap_smooth_regular_16bpc);
    ctx->mct[7] = HWY_DYNAMIC_POINTER(prep_8tap_smooth_16bpc);
    ctx->mct[8] = HWY_DYNAMIC_POINTER(prep_8tap_smooth_sharp_16bpc);
    ctx->mct[9] = HWY_DYNAMIC_POINTER(prep_bilin_16bpc);
    ctx->mc_scaled[0] = HWY_DYNAMIC_POINTER(put_8tap_regular_scaled_16bpc);
    ctx->mc_scaled[1] = HWY_DYNAMIC_POINTER(put_8tap_regular_smooth_scaled_16bpc);
    ctx->mc_scaled[2] = HWY_DYNAMIC_POINTER(put_8tap_regular_sharp_scaled_16bpc);
    ctx->mc_scaled[3] = HWY_DYNAMIC_POINTER(put_8tap_sharp_regular_scaled_16bpc);
    ctx->mc_scaled[4] = HWY_DYNAMIC_POINTER(put_8tap_sharp_smooth_scaled_16bpc);
    ctx->mc_scaled[5] = HWY_DYNAMIC_POINTER(put_8tap_sharp_scaled_16bpc);
    ctx->mc_scaled[6] = HWY_DYNAMIC_POINTER(put_8tap_smooth_regular_scaled_16bpc);
    ctx->mc_scaled[7] = HWY_DYNAMIC_POINTER(put_8tap_smooth_scaled_16bpc);
    ctx->mc_scaled[8] = HWY_DYNAMIC_POINTER(put_8tap_smooth_sharp_scaled_16bpc);
    ctx->mc_scaled[9] = HWY_DYNAMIC_POINTER(put_bilin_scaled_16bpc);
    ctx->mct_scaled[0] = HWY_DYNAMIC_POINTER(prep_8tap_regular_scaled_16bpc);
    ctx->mct_scaled[1] = HWY_DYNAMIC_POINTER(prep_8tap_regular_smooth_scaled_16bpc);
    ctx->mct_scaled[2] = HWY_DYNAMIC_POINTER(prep_8tap_regular_sharp_scaled_16bpc);
    ctx->mct_scaled[3] = HWY_DYNAMIC_POINTER(prep_8tap_sharp_regular_scaled_16bpc);
    ctx->mct_scaled[4] = HWY_DYNAMIC_POINTER(prep_8tap_sharp_smooth_scaled_16bpc);
    ctx->mct_scaled[5] = HWY_DYNAMIC_POINTER(prep_8tap_sharp_scaled_16bpc);
    ctx->mct_scaled[6] = HWY_DYNAMIC_POINTER(prep_8tap_smooth_regular_scaled_16bpc);
    ctx->mct_scaled[7] = HWY_DYNAMIC_POINTER(prep_8tap_smooth_scaled_16bpc);
    ctx->mct_scaled[8] = HWY_DYNAMIC_POINTER(prep_8tap_smooth_sharp_scaled_16bpc);
    ctx->mct_scaled[9] = HWY_DYNAMIC_POINTER(prep_bilin_scaled_16bpc);
    ctx->warp8x8 = HWY_DYNAMIC_POINTER(warp_affine_8x8_16bpc);
    ctx->warp8x8t = HWY_DYNAMIC_POINTER(warp_affine_8x8t_16bpc);
}

}  // namespace dav1d

extern "C" void dav1d_mc_dsp_init_hwy_8bpc(void *const c) {
    dav1d::hwy_init_chosen_target();
    dav1d::mc_dsp_init_8bpc_hwy(c);
}

extern "C" void dav1d_mc_dsp_init_hwy_16bpc(void *const c) {
    dav1d::hwy_init_chosen_target();
    dav1d::mc_dsp_init_16bpc_hwy(c);
}

#endif  // HWY_ONCE
