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

// Palette index packing (src/pal.c) implemented with Google Highway: one
// source is compiled per SIMD target and the best one supported by the CPU
// is selected at runtime (HWY_DYNAMIC_DISPATCH). Bit-exact with the C code.

#include <stdint.h>
#include <string.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/hwy/pal.cpp"
#include "hwy/foreach_target.h"

#include "hwy/highway.h"
#include "src/hwy/common.h"

HWY_BEFORE_NAMESPACE();

namespace dav1d {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

using D16 = hn::CappedTag<uint8_t, 16>;

// All pack variants compute out[i] = src[2*i] | (src[2*i+1] << 4), truncating
// like the C implicit conversion (Or/ShiftLeft, no saturation). Every load
// stays within the w source bytes the C code reads.

// 32 src bytes -> 16 dst bytes.
static inline hn::VFromD<D16> hwy_pal_pack16(const uint8_t *const src,
                                             const D16 d)
{
    hn::VFromD<D16> v0, v1;
    hn::LoadInterleaved2(d, src, v0, v1);
    return hn::Or(v0, hn::ShiftLeft<4>(v1));
}

// 16 src bytes -> 8 dst bytes.
static inline void hwy_pal_pack8(uint8_t *const dst, const uint8_t *const src,
                                 const D16 d)
{
    const auto v = hn::LoadU(d, src);
    const auto p = hn::Or(hn::ConcatEven(d, v, v),
                          hn::ShiftLeft<4>(hn::ConcatOdd(d, v, v)));
    hn::StoreN(p, d, dst, 8);
}

// 8 src bytes -> 4 dst bytes.
static inline void hwy_pal_pack4(uint8_t *const dst, const uint8_t *const src)
{
    const hn::CappedTag<uint8_t, 8> d8;
    const auto v = hn::LoadU(d8, src);
    const auto p = hn::Or(hn::ConcatEven(d8, v, v),
                          hn::ShiftLeft<4>(hn::ConcatOdd(d8, v, v)));
    hn::StoreN(p, d8, dst, 4);
}

// Port of pal_idx_finish_c. bw/bh are powers of two in [4, 64], w/h multiples
// of 4 with 4 <= w <= bw, 4 <= h <= bh. All overlapping tails stay within the
// w bytes the C code reads and rewrite identical values.
void hwy_pal_idx_finish(uint8_t *dst, const uint8_t *src,
                        const int bw, const int bh, const int w, const int h)
{
    const int dst_w = w / 2, dst_bw = bw / 2;
    const D16 d16;

    if (w == bw) { // full-width block (the hot case): no horizontal padding
        switch (dst_w) {
        case 32:
            for (int y = 0; y < h; y++, src += bw, dst += 32) {
                hn::StoreU(hwy_pal_pack16(src, d16), d16, dst);
                hn::StoreU(hwy_pal_pack16(src + 32, d16), d16, dst + 16);
            }
            break;
        case 16:
            for (int y = 0; y < h; y++, src += bw, dst += 16)
                hn::StoreU(hwy_pal_pack16(src, d16), d16, dst);
            break;
        case 8:
            for (int y = 0; y < h; y++, src += bw, dst += 8)
                hwy_pal_pack8(dst, src, d16);
            break;
        case 4:
            for (int y = 0; y < h; y++, src += bw, dst += 4)
                hwy_pal_pack4(dst, src);
            break;
        default: // dst_w == 2: too narrow for a safe 8-byte load
            for (int y = 0; y < h; y++, src += bw, dst += 2) {
                dst[0] = src[0] | (src[1] << 4);
                dst[1] = src[2] | (src[3] << 4);
            }
            break;
        }
    } else if (dst_w >= 16) { // w >= 32
        for (int y = 0; y < h; y++, src += bw, dst += dst_bw) {
            int x = 0;
            for (; x + 16 <= dst_w; x += 16)
                hn::StoreU(hwy_pal_pack16(src + 2 * x, d16), d16, dst + x);
            if (x < dst_w)
                hn::StoreU(hwy_pal_pack16(src + 2 * (dst_w - 16), d16), d16,
                           dst + dst_w - 16);
            memset(dst + dst_w, src[w - 1] * 0x11, dst_bw - dst_w);
        }
    } else if (dst_w >= 8) { // w >= 16
        for (int y = 0; y < h; y++, src += bw, dst += dst_bw) {
            hwy_pal_pack8(dst, src, d16);
            if (dst_w > 8)
                hwy_pal_pack8(dst + dst_w - 8, src + 2 * (dst_w - 8), d16);
            memset(dst + dst_w, src[w - 1] * 0x11, dst_bw - dst_w);
        }
    } else if (dst_w >= 4) { // w >= 8
        for (int y = 0; y < h; y++, src += bw, dst += dst_bw) {
            hwy_pal_pack4(dst, src);
            if (dst_w > 4) // dst_w == 6
                hwy_pal_pack4(dst + 2, src + 4);
            memset(dst + dst_w, src[w - 1] * 0x11, dst_bw - dst_w);
        }
    } else { // dst_w == 2 (w == 4)
        for (int y = 0; y < h; y++, src += bw, dst += dst_bw) {
            dst[0] = src[0] | (src[1] << 4);
            dst[1] = src[2] | (src[3] << 4);
            memset(dst + 2, src[w - 1] * 0x11, dst_bw - 2);
        }
    }

    if (h < bh) {
        const uint8_t *const last_row = dst - dst_bw;
        for (int y = h; y < bh; y++, dst += dst_bw)
            memcpy(dst, last_row, dst_bw);
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace dav1d

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace dav1d {
HWY_EXPORT(hwy_pal_idx_finish);
}  // namespace dav1d

namespace {
// Mirror of Dav1dPalDSPContext (src/pal.h).
using PalIdxFinishFn = void (*)(uint8_t *, const uint8_t *, int, int, int, int);
struct PalDSP {
    PalIdxFinishFn pal_idx_finish;
};
}  // namespace

namespace dav1d {

static void pal_dsp_init_hwy(void *const c) {
    auto *const ctx = static_cast<PalDSP *>(c);
    ctx->pal_idx_finish = HWY_DYNAMIC_POINTER(hwy_pal_idx_finish);
}

}  // namespace dav1d

extern "C" void dav1d_pal_dsp_init_hwy(void *const c) {
    dav1d::hwy_init_chosen_target();
    dav1d::pal_dsp_init_hwy(c);
}

#endif  // HWY_ONCE
