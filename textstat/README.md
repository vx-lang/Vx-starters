# textstat

Counts the words, numbers and runs of capital letters in a file, using the
[`textkit`](../textkit) library.

```
$ ./build.sh
ok:
words 25
numbers 3
upper 8
```

Then `build/textstat <file>` counts any file.

`build.sh` checks the result against `wc -w` and `grep -o`. It finds textkit by putting the
repository root on the import path (`VX_STD_PATH`); see textkit's README.
