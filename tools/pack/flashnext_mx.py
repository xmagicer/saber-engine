"""MXFP4 helpers (numpy only), shared by the Flash-Next and Quark 27B builders and checkers.

Two INDEPENDENT dequantizations of the same weight, compared bit for bit:

* ``quark_dequant`` — the checkpoint's own dequant, ported line for line from AMD Quark's torch
  reference for MX formats (``upcast_from_mxfp`` + its e2m1 -> fp32 table): evens = ``x & 0xF``
  (the LOW nibble is element 2i), the table ``[0, .5, 1, 1.5, 2, 3, 4, 6]`` then its negation, the
  scale as the fp32 whose exponent field is the e8m0 byte (``(s << 23).view(float32)``), applied
  per 32 along K. It reads the CHECKPOINT's bytes (``[N, K/2]`` nibbles, ``[N, K/32]`` scales).
* ``saber_dequant`` — Saber's reading of the PACK's bytes: undo ``permute_w`` (WMMA fragment order,
  ``build_pack.py``), read the scales transposed ``[K/32, N]``, decode with ``build_pack.py``'s
  e2m1 rule (code & 7 on the e2m1 grid, sign from bit 3) and scale by ``2 ** (e - 127)``.

The two agree bit for bit iff the pack's bytes are a faithful re-layout of the checkpoint's AND
the nibble order is low-first. ``nibble_swap_control`` is the negative control (it must NOT agree).
"""

from __future__ import annotations

import numpy as np

MX_BLOCK = 32
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
# Quark's table: outputs = cat([vals, -vals]) indexed by the 4-bit code.
QUARK_TABLE = np.concatenate([E2M1, -E2M1]).astype(np.float32)


def permute_w(packed: np.ndarray, n: int, k: int) -> np.ndarray:
    """[N, K/2] -> WMMA fragment order (as ``build_pack.py`` lays it out). A byte permutation."""
    if n % 16 or k % 16:
        raise ValueError(f"permute_w needs N, K % 16 == 0, got N={n} K={k}")
    nt, ks = n // 16, k // 16
    return packed.reshape(nt, 16, ks, 2, 4).transpose(0, 2, 3, 1, 4).reshape(n, k // 2)


def unpermute_w(perm: np.ndarray, n: int, k: int) -> np.ndarray:
    """Inverse of ``permute_w``."""
    nt, ks = n // 16, k // 16
    return perm.reshape(nt, ks, 2, 16, 4).transpose(0, 3, 1, 2, 4).reshape(n, k // 2)


def quark_dequant(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Quark's ``upcast_from_mxfp_torch`` over CHECKPOINT bytes: w [N, K/2] u8, s [N, K/32] u8."""
    x = w.astype(np.int32)
    evens = x & 0xF
    odds = (x >> 4) & 0xF
    even_f = QUARK_TABLE[evens]
    odd_f = QUARK_TABLE[odds]
    fp32 = np.stack([even_f, odd_f], axis=-1).reshape(w.shape[0], -1)
    dq_scale = (s.astype(np.int32) << 23).view(np.float32)
    n, k = fp32.shape
    return (fp32.reshape(n, k // MX_BLOCK, MX_BLOCK) * dq_scale[:, :, None]).reshape(n, k)


def saber_dequant(w_perm: np.ndarray, ws_t: np.ndarray, n: int, k: int) -> np.ndarray:
    """Saber's reading of PACK bytes: w_perm [N, K/2] fragment order, ws_t [K/32, N]."""
    packed = unpermute_w(w_perm, n, k)
    codes = np.stack([packed & 0x0F, packed >> 4], axis=-1).reshape(n, k)
    mag = E2M1[codes & 0x7]
    val = np.where((codes & 0x8) != 0, -mag, mag).astype(np.float32)
    scale = np.exp2(ws_t.T.astype(np.float32) - 127.0).astype(np.float32)
    return (val.reshape(n, k // MX_BLOCK, MX_BLOCK) * scale[:, :, None]).reshape(n, k)


def nibble_swap_control(w: np.ndarray) -> np.ndarray:
    """The negative control: the same bytes with the two nibbles of every byte swapped."""
    return ((w & 0x0F) << 4) | (w >> 4)


def bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def scale_scan(s: np.ndarray, hist: np.ndarray, spread: np.ndarray) -> None:
    """The e8m0 scan, accumulated: value histogram (256 bins) and per-row spread (max - min)."""
    hist += np.bincount(s.reshape(-1), minlength=256).astype(np.int64)
    sp = s.max(axis=1).astype(np.int64) - s.min(axis=1).astype(np.int64)
    spread += np.bincount(sp, minlength=256).astype(np.int64)
