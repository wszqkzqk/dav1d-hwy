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

// Temporal motion vector DSP functions (src/refmvs.c) implemented with Google
// Highway: one source is compiled per SIMD target and the best one supported
// by the CPU is selected at runtime (HWY_DYNAMIC_DISPATCH). Bit-exact with
// the C code.

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/hwy/refmvs.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "src/hwy/common.h"

// Block width/height in 4x4 units, defined in src/tables.c.
extern "C" const uint8_t dav1d_block_dimensions[][4];

HWY_BEFORE_NAMESPACE();

namespace dav1d {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Byte offsets into the packed 12-byte refmvs_block (src/refmvs.h):
// [0..3] mv.mv[0].n, [4..7] mv.mv[1].n, [8] ref.ref[0], [9] ref.ref[1],
// [10] bs, [11] mf.

// Port of splat_mv_c. bw4 is a power of two in [1, 32]; every store covers
// exactly the bytes the C loop writes.
void hwy_splat_mv(uint8_t **rr, const uint8_t *const rmv,
                         const int bx4, const int bw4, int bh4)
{
    const size_t off = (size_t) bx4 * 12;
    if (bw4 <= 2) {
        // Scalar stores, no pattern setup. For bw4 == 2, three u64 stores per
        // row: a covers record bytes [0, 8), c1 the [8, 16) phase (rec[8..11],
        // rec[0..3]), c2 the [16, 24) phase.
        uint64_t a;
        uint32_t b;
        memcpy(&a, rmv, 8);
        memcpy(&b, rmv + 8, 4);
        const uint64_t c1 = b | (a << 32);
        const uint64_t c2 = (a >> 32) | ((uint64_t) b << 32);
        do {
            uint8_t *const r = *rr++ + off;
            memcpy(r, &a, 8);
            if (bw4 == 1) {
                memcpy(r + 8, &b, 4);
            } else {
                memcpy(r + 8, &c1, 8);
                memcpy(r + 16, &c2, 8);
            }
        } while (--bh4);
        return;
    }
    // 48 bytes = lcm(12, 16): the 12-byte record tiled to three full vectors.
    alignas(16) uint8_t pat[48];
    for (int i = 0; i < 48; i += 12) memcpy(pat + i, rmv, 12);
    const hn::CappedTag<uint8_t, 16> d;
    const auto v0 = hn::Load(d, pat);
    const auto v1 = hn::Load(d, pat + 16);
    const auto v2 = hn::Load(d, pat + 32);
    const int groups = bw4 >> 2; // 48 bytes per 4 blocks
    do {
        uint8_t *r = *rr++ + off;
        for (int k = 0; k < groups; k++, r += 48) {
            hn::StoreU(v0, d, r);
            hn::StoreU(v1, d, r + 16);
            hn::StoreU(v2, d, r + 32);
        }
    } while (--bh4);
}

// (abs(mv.y) | abs(mv.x)) < 4096 on a packed mv (.n = y | x << 16).
static inline bool hwy_mv_small(const uint32_t n)
{
    const int y = (int16_t) n;
    const int x = (int16_t) (n >> 16);
    const unsigned ay = (unsigned) (y < 0 ? -y : y);
    const unsigned ax = (unsigned) (x < 0 ? -x : x);
    return (ay | ax) < 4096;
}

// Expands the 5-byte record in vsrc lanes [0, 5) to the 16-byte chunk of the
// infinite periodic pattern whose first byte has phase p: kSplat5Phase[p][i]
// = (p + i) % 5. Chunk at byte offset off has phase off % 5.
alignas(16) static const uint8_t kSplat5Phase[5][16] = {
#define SPLAT5_ROW(p) { (p + 0) % 5, (p + 1) % 5, (p + 2) % 5, (p + 3) % 5, \
    (p + 4) % 5, (p + 5) % 5, (p + 6) % 5, (p + 7) % 5, (p + 8) % 5, \
    (p + 9) % 5, (p + 10) % 5, (p + 11) % 5, (p + 12) % 5, (p + 13) % 5, \
    (p + 14) % 5, (p + 15) % 5 }
    SPLAT5_ROW(0), SPLAT5_ROW(1), SPLAT5_ROW(2), SPLAT5_ROW(3), SPLAT5_ROW(4)
#undef SPLAT5_ROW
};

// Writes bw8 copies (bw8 in {1, 2, 4, 8, 16}) of the 5-byte record packed in
// the low bytes of rec, i.e. exactly bw8 * 5 bytes at dst.
template <class D>
static inline void hwy_splat5(uint8_t *const dst, const uint64_t rec,
                              const int bw8, const D d,
                              const hn::Vec<D> *const vph)
{
    if (bw8 == 1) {
        const uint32_t lo = (uint32_t) rec;
        memcpy(dst, &lo, 4);
        dst[4] = (uint8_t) (rec >> 32);
        return;
    }
    if (bw8 == 2) {
        const uint64_t rec8 = rec | (rec << 40); // bytes [0, 10) of the pattern
        const uint16_t hi = (uint16_t) (rec >> 24); // rec[3], rec[4]
        memcpy(dst, &rec8, 8);
        memcpy(dst + 8, &hi, 2);
        return;
    }
    // Only lanes [0, 5) of vsrc are referenced by the phase tables.
    uint8_t tmp[16];
    memcpy(tmp, &rec, 5);
    const auto vsrc = hn::LoadU(d, tmp);
    const int n = bw8 * 5;
    int off = 0;
    for (int p = 0; off + 16 <= n; off += 16, p = p == 4 ? 0 : p + 1)
        hn::StoreU(hn::TableLookupBytes(vsrc, vph[p]), d, dst + off);
    // n % 16 is 4 or 8; the overlapping tail at n - 16 has phase 4 because
    // n == 0 (mod 5). Overwritten bytes carry identical pattern values.
    if (off < n)
        hn::StoreU(hn::TableLookupBytes(vsrc, vph[4]), d, dst + n - 16);
}

// Port of save_tmvs_c; rp/rr index refmvs_temporal_block (5 bytes) and
// refmvs_block (12 bytes) arrays, accessed here at byte level.
void hwy_save_tmvs(uint8_t *rp, const ptrdiff_t stride,
                          uint8_t *const *const rr,
                          const uint8_t *const ref_sign,
                          const int col_end8, const int row_end8,
                          const int col_start8, const int row_start8)
{
    const hn::CappedTag<uint8_t, 16> d;
    const hn::Vec<decltype(d)> vph[5] = {
        hn::Load(d, kSplat5Phase[0]), hn::Load(d, kSplat5Phase[1]),
        hn::Load(d, kSplat5Phase[2]), hn::Load(d, kSplat5Phase[3]),
        hn::Load(d, kSplat5Phase[4]),
    };
    const ptrdiff_t stride5 = stride * 5;
    for (int y = row_start8; y < row_end8; y++) {
        const uint8_t *const b = rr[(y & 15) * 2];
        int x = col_start8;
        while (x < col_end8) {
            const uint8_t *const cand = b + (size_t) (x * 2 + 1) * 12;
            const int bw8 = (dav1d_block_dimensions[cand[10]][0] + 1) >> 1;
            uint32_t mv0, mv1;
            memcpy(&mv0, cand, 4);
            memcpy(&mv1, cand + 4, 4);
            const int ref0 = (int8_t) cand[8];
            const int ref1 = (int8_t) cand[9];
            uint64_t rec;
            if (ref1 > 0 && ref_sign[ref1 - 1] && hwy_mv_small(mv1)) {
                rec = mv1 | ((uint64_t) cand[9] << 32);
            } else if (ref0 > 0 && ref_sign[ref0 - 1] && hwy_mv_small(mv0)) {
                rec = mv0 | ((uint64_t) cand[8] << 32);
            } else {
                rec = 0;
            }
            hwy_splat5(rp + (size_t) x * 5, rec, bw8, d, vph);
            x += bw8;
        }
        rp += stride5;
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace dav1d

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace dav1d {
HWY_EXPORT(hwy_splat_mv);
HWY_EXPORT(hwy_save_tmvs);
}  // namespace dav1d

namespace {
// Mirror of Dav1dRefmvsDSPContext (src/refmvs.h), so that this file does not
// need dav1d's C headers. Pointer types differ but are ABI-compatible.
using LoadTmvsFn = void (*)(const void *, int, int, int, int, int);
using SaveTmvsFn = void (*)(uint8_t *, ptrdiff_t, uint8_t *const *,
                            const uint8_t *, int, int, int, int);
using SplatMvFn = void (*)(uint8_t **, const uint8_t *, int, int, int);
struct RefmvsDSP {
    LoadTmvsFn load_tmvs;
    SaveTmvsFn save_tmvs;
    SplatMvFn splat_mv;
};
}  // namespace

namespace dav1d {

static void refmvs_dsp_init_hwy(void *const c) {
    auto *const ctx = static_cast<RefmvsDSP *>(c);
    ctx->save_tmvs = HWY_DYNAMIC_POINTER(hwy_save_tmvs);
    ctx->splat_mv = HWY_DYNAMIC_POINTER(hwy_splat_mv);
}

}  // namespace dav1d

extern "C" void dav1d_refmvs_dsp_init_hwy(void *const c) {
    dav1d::hwy_init_chosen_target();
    dav1d::refmvs_dsp_init_hwy(c);
}

#endif  // HWY_ONCE
