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

// Helpers shared by the Google Highway backends (src/hwy/*.cpp). No include
// guard on purpose: the including source is re-included once per SIMD target
// by hwy/foreach_target.h and every pass needs its own copy of these
// definitions inside that target's HWY_NAMESPACE. Include after
// hwy/highway.h.

#include <stddef.h>
#include <stdint.h>
#include <type_traits>

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

// imin/imax/iabs match include/common/intops.h.
static inline int hwy_imin(const int a, const int b) { return a < b ? a : b; }
static inline int hwy_imax(const int a, const int b) { return a > b ? a : b; }
static inline int hwy_iabs(const int v) { return v < 0 ? -v : v; }

// Matches ctz() in include/common/intops.h. dc_gen() divides by ctz of sums
// like 4+8 that are not powers of two, so ulog2 would not be equivalent.
static inline int hwy_ctz(const unsigned v) {
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long idx;
    _BitScanForward(&idx, v);
    return (int) idx;
#else
    return __builtin_ctz(v);
#endif
}

// Loads n values as i16 lanes; lanes [n, Lanes) are zero. Source pixels are
// <= 4095, so the u16->i16 bitcasts are exact. kFull selects unconditional
// full-vector accesses: dav1d block widths >= 8 are multiples of 8, so the
// partial (LoadN) path only ever sees w < 8.
template <int kCap, class D16, typename Src>
static HWY_INLINE hn::VFromD<D16> LoadI16(const D16 d, const Src *const p,
                                          const size_t n) {
    constexpr bool kFull = kCap != 0;
    if constexpr (sizeof(Src) == 1) {
        const hn::Rebind<uint8_t, D16> d8;
        const hn::Rebind<uint16_t, D16> du16;
        const auto v = kFull ? hn::LoadU(d8, p) : hn::LoadN(d8, p, n);
        return hn::BitCast(d, hn::PromoteTo(du16, v));
    } else if constexpr (std::is_same<Src, int16_t>::value) {
        return kFull ? hn::LoadU(d, p) : hn::LoadN(d, p, n);
    } else {
        const hn::Rebind<uint16_t, D16> du16;
        const auto v = kFull ? hn::LoadU(du16, p) : hn::LoadN(du16, p, n);
        return hn::BitCast(d, v);
    }
}

// Stores i16 lanes as pixels; v must already be clamped to [0, bitdepth_max],
// so the narrowing is exact.
template <int kCap, class D16, typename Pixel>
static HWY_INLINE void StorePx(const D16, Pixel *const p, const size_t n,
                               const hn::VFromD<D16> v) {
    constexpr bool kFull = kCap != 0;
    const hn::Rebind<Pixel, D16> dp;
    if constexpr (sizeof(Pixel) == 1) {
        const auto v8 = hn::DemoteTo(dp, v);
        if (kFull) hn::StoreU(v8, dp, p); else hn::StoreN(v8, dp, p, n);
    } else {
        const auto v16 = hn::BitCast(dp, v);
        if (kFull) hn::StoreU(v16, dp, p); else hn::StoreN(v16, dp, p, n);
    }
}

// i16 -> i32 of the sequential lower/upper half (column order, unlike
// MulEven/MulOdd).
template <class D32, class V16>
static HWY_INLINE hn::VFromD<D32> WidenLo(const D32 d32, const V16 v) {
    return hn::PromoteTo(d32, hn::LowerHalf(v));
}
template <class D32, class V16>
static HWY_INLINE hn::VFromD<D32> WidenHi(const D32 d32, const V16 v) {
    return hn::PromoteUpperTo(d32, v);
}

// Packs i32 lo/hi half results back into i16 column order; the narrowing
// demote saturates (the callers hold in-range values or clamp afterwards).
template <class D16, class V32>
static HWY_INLINE hn::VFromD<D16> PackHalves(const D16 d, const V32 lo,
                                             const V32 hi) {
    const hn::Rebind<int16_t, hn::DFromV<V32>> dh;
    return hn::Combine(d, hn::DemoteTo(dh, hi), hn::DemoteTo(dh, lo));
}

// Widens pixel lanes (u8/u16, values <= 4095) to int32 lanes.
template <typename Pixel, class D32>
static inline hn::VFromD<D32> hwy_load_px(const D32 d32, const Pixel *const p,
                                          const int n)
{
    const hn::Rebind<Pixel, D32> dp;
    const int L = (int) hn::Lanes(d32);
    const auto v = n >= L ? hn::LoadU(dp, p) : hn::LoadN(dp, p, n);
    if constexpr (sizeof(Pixel) == 1) {
        const hn::Rebind<uint16_t, D32> du16;
        return hn::PromoteTo(d32, hn::PromoteTo(du16, v));
    } else {
        return hn::PromoteTo(d32, v);
    }
}

// Stores int32 lanes as pixels; the caller has clamped to [0, bitdepth_max],
// so the narrowing demotions are exact.
template <typename Pixel, class D32>
static inline void hwy_store_px(Pixel *const p, const D32 d32,
                                const hn::VFromD<D32> v, const int n)
{
    const hn::Rebind<Pixel, D32> dp;
    const int L = (int) hn::Lanes(d32);
    if constexpr (sizeof(Pixel) == 1) {
        const hn::Rebind<uint16_t, D32> du16;
        const auto v8 = hn::DemoteTo(dp, hn::DemoteTo(du16, v));
        if (n >= L) hn::StoreU(v8, dp, p); else hn::StoreN(v8, dp, p, n);
    } else {
        const auto v16 = hn::DemoteTo(dp, v);
        if (n >= L) hn::StoreU(v16, dp, p); else hn::StoreN(v16, dp, p, n);
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace dav1d

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace dav1d {

// Resolve the best per-target function pointers once at init; the
// ChosenTarget must be initialized first or the tables yield their
// re-dispatching first entry.
static void hwy_init_chosen_target() {
    hwy::GetChosenTarget().Update(hwy::SupportedTargets());
}

}  // namespace dav1d

#endif  // HWY_ONCE
