# safetensors

Write and read `.safetensors`, the checkpoint format most model weights ship in today.

```
$ ./build.sh
embed.weight [4, 3]: -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.25 1.5 1.75
head.bias [3]: 0.5 -1.5 3
steps [1]: 1234567890123
broken files refused for the right reason: 6 of 6
ok
ok: Python's json and struct read the same tensors
```

A `.safetensors` file has three parts:

1. 8 bytes: the header's length, a little-endian `u64`.
2. The header: a JSON object giving each tensor's `dtype`, `shape` and `data_offsets`
   `[begin, end)`, plus an optional `__metadata__` object of strings. It is padded with spaces.
3. The data: every tensor's bytes, little-endian, one after another with no gaps.

`src/main.vx` writes a small checkpoint: a 4 x 3 `F32` matrix, a 3-element `F32` vector and one
`I64`. Then it reads the file back with its own small JSON parser and checks everything a reader
must check:

- the header fits in the file;
- the dtype is known;
- each data range holds exactly `prod(shape) x dtype size` bytes;
- the ranges, taken in order, cover the data with no gap and no overlap, and end where the file
  ends.

It prints the tensors and compares every value with what it wrote. Then it gives the reader six
broken files, and each must be refused for its own reason:

- a header length past the end of the file;
- a range one element too long;
- an unknown dtype;
- a missing brace;
- an extra byte after the data;
- a missing last byte.

`build.sh` then reads the same file with `check.py`, which uses only Python's `json` and `struct`,
and compares what the two readers print. That catches mistakes the Vx program would make the same
way when writing and reading. Writing and reading big-endian, for example, passes every check in
Vx, and only Python sees the bad file.

Each of these changes makes it fail:

- big-endian on both sides (caught by Python);
- not checking where the data ends;
- `F32` taken as 8 bytes;
- a gap between two tensors in the header;
- not recognizing `__metadata__`;
- name matching that ignores the characters.

## Notes

- `std::fs` cannot tell a failed open from an empty file, so a missing file reads as zero bytes
  (see [FRICTION.md](../FRICTION.md)), and the reader then refuses it because the header length
  is missing.
- One parser function, `expect`, ends in `return;`. Without it the whole program falls back to
  the older code generator ([vx-lang/Vx#1651](https://github.com/vx-lang/Vx/issues/1651)).

Needs Vx newer than v0.0.3 (built from `main`), and `python3`; see the
[top-level README](../README.md).
