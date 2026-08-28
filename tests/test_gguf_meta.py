"""The minimal GGUF reader that backs the probe when the `gguf` package fails.

Synthetic GGUF files exercise both metadata layouts (unaligned and 32-byte
aligned values) so the fallback cannot silently depend on the layout of the
Bonsai artefact it was written for.
"""

from __future__ import annotations

import struct

import pytest

from harness import gguf_meta

_STR, _U32, _ARRAY, _BOOL = 8, 4, 9, 7


def _s(kv: list[tuple[str, tuple[int, bytes]]], names: list[str], align: bool):
    chunks = [b"GGUF", struct.pack("<I", 3), struct.pack("<Q", len(names)),
              struct.pack("<Q", len(kv))]

    def pad_value(block: bytes) -> bytes:
        if not align:
            return block
        # Absolute alignment: the next key starts on a 32-byte boundary, which
        # is exactly the rule the reader applies after each value.
        offset = sum(len(c) for c in chunks)
        padded = (offset + len(block)) % 32
        return block + b"\x00" * ((32 - padded) % 32)

    for key, (value_type, value) in kv:
        key_bytes = key.encode()
        chunks.append(struct.pack("<Q", len(key_bytes)) + key_bytes)
        chunks.append(struct.pack("<I", value_type))
        chunks.append(pad_value(value))
    for name in names:
        name_bytes = name.encode()
        chunks.append(struct.pack("<Q", len(name_bytes)) + name_bytes)
        chunks.append(struct.pack("<I", 1))  # n_dims
        chunks.append(struct.pack("<Q", 4))  # dims[0]
        chunks.append(struct.pack("<I", 0))  # tensor type f32
        chunks.append(struct.pack("<q", 0))  # offset
    return b"".join(chunks)


def _string(value: str) -> bytes:
    return struct.pack("<Q", len(value)) + value.encode("utf-8")


def _lfm2_metadata() -> list[tuple[str, tuple[int, bytes]]]:
    return [
        ("general.architecture", (_STR, _string("lfm2"))),
        ("lfm2.block_count", (_U32, struct.pack("<I", 30))),
        ("lfm2.attention.head_count", (_U32, struct.pack("<I", 32))),
        ("lfm2.attention.head_count_kv", (_U32, struct.pack("<I", 8))),
        ("lfm2.embedding_length", (_U32, struct.pack("<I", 2048))),
        ("lfm2.context_length", (_U32, struct.pack("<I", 131072))),
        ("lfm2.rope.freq_base", (_U32, struct.pack("<I", 10000))),
        # An array of strings, as real vocab metadata is.
        ("tokenizer.ggml.model", (_ARRAY, struct.pack("<I", _STR) +
                                  struct.pack("<Q", 2) +
                                  _string("alpha") + _string("beta"))),
    ]


def _tensor_names() -> list[str]:
    # 30 blocks; attention at the LFM2 set {2, 5, 9, 13, 17, 21, 24, 27}.
    attention = {2, 5, 9, 13, 17, 21, 24, 27}
    names = ["token_embd.weight"]
    for index in range(30):
        names.append(f"blk.{index}.attn_norm.weight")
        if index in attention:
            names.append(f"blk.{index}.attn_q.weight")
            names.append(f"blk.{index}.attn_k.weight")
    return names


@pytest.mark.parametrize("align", [False, True])
def test_reads_metadata_and_tensor_names(tmp_path, align):
    payload = _s(_lfm2_metadata(), _tensor_names(), align)
    path = tmp_path / "model.gguf"
    path.write_bytes(payload)

    version, count, metadata, names = gguf_meta.parse(path)

    assert version == 3
    assert count == len(_tensor_names())
    assert metadata["general.architecture"] == "lfm2"
    assert metadata["lfm2.attention.head_count"] == 32
    assert metadata["lfm2.context_length"] == 131072
    assert metadata["tokenizer.ggml.model"] == ["alpha", "beta"]
    assert "blk.2.attn_q.weight" in names
    assert "blk.1.attn_q.weight" not in names


@pytest.mark.parametrize("align", [False, True])
def test_counting_attention_blocks_recovers_the_lfm2_set(tmp_path, align):
    payload = _s(_lfm2_metadata(), _tensor_names(), align)
    path = tmp_path / "model.gguf"
    path.write_bytes(payload)

    _, _, _, names = gguf_meta.parse(path)
    blocks = {
        int(m.group(1))
        for name in names
        if (m := gguf_meta_re.match(r"blk\.(\d+)\.attn_q\.weight", name))
    }
    assert blocks == {2, 5, 9, 13, 17, 21, 24, 27}


def test_non_gguf_files_raise(tmp_path):
    path = tmp_path / "not.gguf"
    path.write_bytes(b"definitely not a gguf file")
    with pytest.raises(ValueError):
        gguf_meta.parse(path)


# The regex lives with the probe; reaching for it in the test pins the naming
# contract both sides depend on.
import re as gguf_meta_re  # noqa: E402