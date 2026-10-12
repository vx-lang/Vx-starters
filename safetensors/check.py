"""Read a .safetensors file with only json and struct, and print it as src/main.vx does.

A second reader that shares no code with the Vx one: build.sh compares what the two print.
"""
import json
import struct
import sys

SIZES = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "F16": 2, "BF16": 2, "U8": 1}
FORMATS = {"F32": "<f", "I64": "<q"}


def show(x):
    """A number the way Vx's print! writes it: no trailing .0."""
    s = repr(x)
    return s[:-2] if s.endswith(".0") else s


def main(path):
    raw = open(path, "rb").read()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + n].decode("utf-8"))
    data = raw[8 + n :]
    header.pop("__metadata__", None)
    covered = 0
    for name, t in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        begin, end = t["data_offsets"]
        count = 1
        for d in t["shape"]:
            count *= d
        assert begin == covered, f"{name}: hole or overlap"
        assert end - begin == count * SIZES[t["dtype"]], f"{name}: size"
        covered = end
    assert covered == len(data), "data does not end where the last tensor does"
    # In the order of the header, as the Vx reader prints.
    for name, t in header.items():
        begin, end = t["data_offsets"]
        fmt = FORMATS[t["dtype"]]
        size = struct.calcsize(fmt)
        values = [struct.unpack(fmt, data[i : i + size])[0] for i in range(begin, end, size)]
        dims = ", ".join(str(d) for d in t["shape"])
        print(f"{name} [{dims}]: " + " ".join(show(v) for v in values))


main(sys.argv[1])
