// A C program that calls the Vx function parallel_batch_check. The Vx side starts its own
// worker tasks, so this program starts no threads. A Rust program calls it the same way.
#include <stdint.h>
#include <stdio.h>

int64_t parallel_batch_check(int32_t n, int32_t accounts, int32_t block, int32_t workers,
                             int64_t seed);

int main(void) {
  int failed = 0;
  int64_t first = 0;
  for (int workers = 1; workers <= 8; workers *= 2) {
    int64_t reruns = parallel_batch_check(2000, 1000, 50, workers, 7);
    printf("from C, %d workers: reruns %lld\n", workers, (long long)reruns);
    if (workers == 1) {
      first = reruns;
    }
    // A wrong balance gives -1. The reruns depend on the batch only, not on the workers.
    if (reruns < 0 || reruns != first) {
      failed++;
    }
  }
  if (failed == 0) {
    printf("ok\n");
  }
  return failed;
}
