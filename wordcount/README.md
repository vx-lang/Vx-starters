# wordcount

Counts the lines, words and bytes in a file, like `wc`.

```console
$ ./build.sh
ok: 4 9 47
$ build/wordcount sample.txt
4 9 47
```

Or run it without building, through the JIT:

```sh
vxc -a run-jit src/main.vx -- sample.txt
```

It reads the file 4096 bytes at a time into a `Tensor<u8, [4096]>`, and counts a word each time
a non-space byte follows a space. The file closes when `file` goes out of scope, because `File`
implements `Drop`.
