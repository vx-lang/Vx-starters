# reshard

Split a weight matrix for tensor parallelism, and reshard a checkpoint from k1 shards to k2
without gathering it first.

```
$ ./build.sh
Warning[W1009]: Unused function parameter 'hi'
rows, 3 -> 5 shards: 7 pieces, 598 of 851 numbers change rank
rows, 4 -> 8 shards: 10 pieces, 736 of 851 numbers change rank
rows, 8 -> 4 shards: 10 pieces, 736 of 851 numbers change rank
split along rows (row-parallel): 8 splits gathered, 64 direct reshards checked
columns, 3 -> 5 shards: 7 pieces, 592 of 851 numbers change rank
columns, 4 -> 8 shards: 8 pieces, 740 of 851 numbers change rank
columns, 8 -> 4 shards: 8 pieces, 740 of 851 numbers change rank
split along columns (column-parallel): 8 splits gathered, 64 direct reshards checked
ok
```

With tensor parallelism each weight matrix is cut into k shards, one per rank. A column-parallel
layer (Megatron's `ColumnParallelLinear`, the first matrix of an MLP) splits it along its columns;
a row-parallel layer (`RowParallelLinear`, the second) along its rows. When k does not divide
the size, the first `n % k` shards get one more, as `torch.tensor_split` does.

A checkpoint saved with k1 shards and loaded with k2 has to be resharded. Gathering the whole
matrix first needs all of it in one place. Resharding directly copies, for each old shard p and
new shard q whose ranges overlap, just that overlap. Going from 4 to 8 shards of 37 rows takes 10
pieces, because 37 rows do not split evenly either way. Along 23 columns the boundaries line up,
and it takes 8.

Each rank's shard sits in its own buffer, rows by columns, as it would in its own memory, so a
row shard and a column shard are laid out differently.

## What is checked

For a 37 x 23 matrix whose numbers are all different, along rows and along columns, and every
k, k1 and k2 from 1 to 8:

- Gathering a split gives back the matrix exactly, and the spare slots of each rank's buffer
  (shards are not all the same size) are never written.
- Resharding directly from k1 to k2 gives exactly what splitting into k2 gives.
- Contracts, proved when the program compiles: the overlap of two ranges starts and ends inside
  both (`later`, `earlier`), and an index inside a range is a valid position within it
  (`local`). `place` has to check the index against the shard before calling `local`, or the
  program does not compile.

Each of these changes makes it fail:

- dropping the remainder from the shard starts (an index runs out of bounds);
- `later` returning the smaller start (stops the compile);
- copying one row or column too few in each piece;
- a column shard using the whole matrix's width as its row length (out of bounds);
- `owner` picking the shard before the right one (the check in `place` panics);
- swapping rows and columns for a row split;
- removing the check in `place` (stops the compile);
- buffers that start at 0 instead of -1, so spare slots look written.

The warning is [vx-lang/Vx#1647](https://github.com/vx-lang/Vx/issues/1647): `hi` appears only in
`local`'s contract.

Needs Vx newer than v0.0.3 (built from `main`); see the [top-level README](../README.md).
