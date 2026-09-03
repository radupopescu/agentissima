"""GGUF metadata and tensor-name reader, used when the `gguf` package cannot.

The `gguf` library parses the whole file eagerly and raises on tensor
quantisation types it does not know. Bonsai's custom ``Q2_0_g64`` quant
(GGML type 42) crashes it in the constructor, with no option to skip tensors.
The metadata section precedes the tensor section, so this minimal reader stops
after the tensor *names* — it reads header, metadata key/value pairs, and
tensor names only, and never interprets tensor payloads or quantisation types.

The fields this benchmark needs are metadata keys and the set of
``blk.<i>.attn_q.weight`` names (§2.1 geometry), so nothing past line ~name
table is ever read.
"""

from __future__ import annotations

import struct
from pathlib import Path

_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARRAY, _U64, _I64, _F64 = range(13)

GGUF_MAGIC = b"GGUF"
# Metadata values are 32-byte aligned in GGUF files that align their data
# sections. The Bonsai artefact does not pad, so both layouts are tolerated:
# parse without padding first, retry with padding if that decodes nonsense.
_ALIGNMENT = 32

_MIN_HEADER = 24  # magic(4) + version(4) + tensor_count(8) + kv_count(8)


# Bounds on what a *valid* file can declare. Their real purpose is to stop a
# misaligned decode: a garbage length read from the wrong offset would otherwise
# ask the stream for gigabytes, and the reader would dutifully consume the whole
# artefact before failing. Both layouts get tried (see `parse`), so a wrong guess
# must fail cheaply.
_MAX_STRING = 64 * 1024**2
_MAX_ARRAY = 8_000_000  # tokenizer vocabularies run to ~10^6 entries


def _read_string(stream: _Stream, off: int) -> tuple[str, int]:
    stream.ensure(off + 8)
    n = struct.unpack_from("<Q", stream.buf, off)[0]
    if n > _MAX_STRING:
        raise struct.error(f"implausible string length {n} at offset {off}")
    off += 8
    stream.ensure(off + n)
    return stream.buf[off : off + n].decode("utf-8", "replace"), off + n


class _Stream:
    """Grows a buffer over the file so a 2.3 GB artefact is never read whole.

    The buffer is a ``bytearray`` extended in place. Rebinding ``bytes`` with
    ``+=`` instead copies everything read so far on every chunk, which is
    quadratic: the metadata section carries the whole tokenizer vocabulary and
    runs to tens of MB, so that cost is paid tens of thousands of times and the
    read takes minutes rather than a second.
    """

    def __init__(self, path: Path) -> None:
        self._file = open(path, "rb")  # noqa: SIM115 - closed in read()
        self.buf = bytearray()

    def ensure(self, length: int) -> None:
        """Guarantee ``length`` bytes are buffered, or fail.

        Every read goes through this. The previous version checked capacity only
        for the 4-byte value type, then decoded strings and arrays of arbitrary
        length against whatever happened to be buffered — so a value straddling
        a chunk boundary raised `unpack_from requires a buffer of at least ...`
        on a perfectly valid file.
        """
        while len(self.buf) < length:
            chunk = self._file.read(1 << 20)
            if not chunk:
                raise struct.error("EOF before GGUF layout was consumed")
            self.buf.extend(chunk)

    def close(self) -> None:
        self._file.close()


def _parse(path: Path, align: int | None) -> tuple[int, int, dict, list[str], int]:
    stream = _Stream(path)
    try:
        stream.ensure(_MIN_HEADER)
        if stream.buf[:4] != GGUF_MAGIC:
            raise ValueError(f"{path} is not a GGUF file")
        version, n_tensors, n_kv = struct.unpack_from("<IQQ", stream.buf, 4)
        off = _MIN_HEADER

        def scalar(fmt: str, width: int, offset: int) -> tuple[object, int]:
            stream.ensure(offset + width)
            return struct.unpack_from(fmt, stream.buf, offset)[0], offset + width

        metadata: dict[str, object] = {}
        for _ in range(n_kv):
            key, off = _read_string(stream, off)
            value_type, off = scalar("<I", 4, off)
            value: object

            if value_type == _STR:
                value, off = _read_string(stream, off)
            elif value_type == _ARRAY:
                stream.ensure(off + 12)
                item_type, count = struct.unpack_from("<IQ", stream.buf, off)
                if count > _MAX_ARRAY:
                    raise struct.error(f"implausible array length {count}")
                off += 12
                items: list[object] = []
                for _ in range(count):
                    if item_type == _STR:
                        item, off = _read_string(stream, off)
                    elif item_type == _F32:
                        item, off = scalar("<f", 4, off)
                    elif item_type in (_U32, _I32):
                        item, off = scalar("<I", 4, off)
                    elif item_type in (_U64, _I64, _F64):
                        item, off = scalar("<Q", 8, off)
                    else:
                        item = None
                    items.append(item)
                value = items
            elif value_type == _F32:
                # Decoded as a float, not as its bit pattern. Reading it with
                # "<I" was harmless while only integer geometry fields were
                # wanted (§2.2), and wrong the moment anything reads a real
                # float: `general.sampling.temp` came back as 1036831949
                # rather than 0.1.
                value, off = scalar("<f", 4, off)
            elif value_type in (_U32, _I32):
                value, off = scalar("<I", 4, off)
            elif value_type in (_U64, _I64):
                value, off = scalar("<Q", 8, off)
            elif value_type == _BOOL:
                value, off = scalar("<?", 1, off)
            elif value_type in (_U8, _I8):
                value, off = scalar("<B", 1, off)
            elif value_type == _F64:
                value, off = scalar("<d", 8, off)
            else:
                value = None
            metadata[key] = value
            if align:
                off = (off + align - 1) & ~(align - 1)

        names: list[str] = []
        for _ in range(n_tensors):
            name, off = _read_string(stream, off)
            names.append(name)
            n_dims, off = scalar("<I", 4, off)
            off += n_dims * 8 + 4 + 8  # dims, ggml type, tensor offset
    finally:
        stream.close()
    return version, n_tensors, metadata, names, n_kv


def parse(path: str | Path) -> tuple[int, int, dict, list[str]]:
    """Return (version, tensor_count, metadata, tensor_names).

    Unparseable files raise ``ValueError``/``struct.error``. The two metadata
    layouts (aligned and unaligned) are both attempted; a result is only
    accepted when it decoded *completely* — every declared key present, none
    of them empty, and a plausible `general.architecture` — so a misaligned
    but non-crashing decode cannot pass silently.

    The key count is what makes that true. Testing `general.architecture`
    alone was not enough: it is the first key in both layouts, so the wrong
    layout can read it correctly and then walk off into padding, returning
    `{"general.architecture": "lfm2", "": 0}` — accepted, and missing
    everything the caller asked for.
    """
    for align in (None, _ALIGNMENT):
        try:
            version, count, metadata, names, declared_kv = _parse(Path(path), align)
            if (
                isinstance(metadata.get("general.architecture"), str)
                and len(metadata) == declared_kv
                and all(metadata)
            ):
                return version, count, metadata, names
        except (struct.error, ValueError):
            continue
    raise ValueError(f"{path} is not a parseable GGUF file")