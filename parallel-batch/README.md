# parallel-batch

A batch of transfers between accounts, run by worker tasks in parallel, and checked against the
same batch run in order. Its shape is the shape of a parallel blockchain executor: the order of
the transactions decides the result, and the workers must not change it.

```console
$ ./build.sh
2000 transfers in blocks of 50, 8 accounts: reruns 1795 1795 1795 1795
2000 transfers in blocks of 50, 1000 accounts: reruns 110 110 110 110
ok
...
from C, 1 workers: reruns 110
...
ok
```

## How it works

`src/batch.vx` runs the batch in blocks of 50 transfers. For each block:

1. A `tasks` block starts the workers. Each worker takes the next transfer index from an
   `AtomicI64` with `fetch_add`, runs the transfer on the balances from before the block, and
   sends the result on a channel. No worker changes a balance.
2. After the `tasks` block ends, one loop goes through the block in order. A transfer is stale
   when an earlier transfer of the block changed one of its two accounts. A stale transfer runs
   again on the current balances. The others keep their first result.

The tasks borrow the batch and the balances from the function around them. They need no `Arc`,
because a `tasks` block ends only after its tasks end.

## What is checked

- Each run with 1, 2, 4 and 8 workers ends with the same balances as the run in order.
- The number of reruns is the same for each number of workers. A first run always sees the
  balances from before its block, so a rerun depends on the batch, not on which worker ran first.
- `build.sh` runs the program with each task on an OS thread (the default), on the shared
  runtime (`VX_TASK_RUNTIME=shared`), and with three seeds (`VX_TASK_SEED`), which fix the order
  of the tasks on one thread.
- `host/host.c` calls `parallel_batch_check` from C. The Vx code starts its own tasks, so the C
  program starts no threads.

Each of these changes makes it fail:

- checking only the `from` account for a stale transfer (wrong balances);
- `fetch_add(2, ..)` in place of `fetch_add(1, ..)`, so that half of the transfers never run
  (wrong balances);
- counting a rerun when results arrive out of order (the reruns differ between worker counts).

`vxc --sanitize=thread` reports no data race.

## Vx in place of tokio

| Rust | Vx |
| --- | --- |
| `std::thread::scope`, `tokio::spawn` | a `tasks` block and `spawn` |
| a tokio multi-thread runtime | `VX_TASK_RUNTIME=shared`, with no change to the program |
| `async fn`, `.await` | a normal function: `recv`, `join` and `lock` wait in place |
| `tokio::sync::mpsc` | `std::sync::channel` |
| `AtomicU64`, `Ordering::Relaxed` | `AtomicI64`, `AtomicOrdering::Relaxed` from `core::sync::atomic` |

[Tasks and channels](https://github.com/vx-lang/Vx/blob/main/www/book/src/tasks.md) in the Vx
book has the details.

## Calling it from C or Rust

A `pub fn` is exported from the object file with its own name. Link the object with
`libvx_std_core`, which runs the tasks, as `build.sh` does. Nothing else is needed.
