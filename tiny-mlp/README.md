# tiny-mlp

The math of a training run, differentiated by Vx's `grad`.

```console
$ ./build.sh
textbook derivatives: sigmoid, tanh, logistic loss, softmax cross-entropy at 20 points
gradient of the XOR loss, 9 weights: grad vs backprop differ by at most 5.6e-17, vs central differences by at most 7.9e-11
step 0: loss 0.7140881141240948
step 300: loss 0.004921288705094844
...
step 1500: loss 0.0003697389104523244
0 xor 0: p = 0.00028351188487309826
0 xor 1: p = 0.9994891458124611
1 xor 0: p = 0.9995924580308524
1 xor 1: p = 0.00027501681196933665
ok
```

(The real output prints the two differences in full.) `src/main.vx` does three things:

1. **Textbook derivatives against `grad`.** At 20 points each, `grad` agrees within 10^-12 with
   the formulas every training framework relies on: the sigmoid's derivative `s (1 - s)`, tanh's
   `1 - tanh^2`, the logistic loss's gradient in the logit `p - y`, and softmax cross-entropy's
   gradient `softmax - one-hot`.
2. **Backpropagation three ways.** For a 2-2-1 network on XOR (tanh hidden units, a logistic
   output, mean logistic loss), the gradient of the loss from `grad` matches backpropagation
   written out by hand to 10^-16, and central finite differences to 10^-10.
3. **Training.** Adam, with the gradient from `grad`, takes the loss from 0.71 to 0.0004 in 1500
   steps, and the network gets all four XOR cases right.

The program returns the number of failed checks. Each of these changes makes it fail: backprop
forgetting the tanh derivative, cross-entropy counting the target twice, a sigmoid with the wrong
slope, and Adam stepping uphill.

## What Vx v0.0.3 allows, and what this does about it

- `grad` moves one scalar, the function's first parameter. `vjp`, reverse mode, would give the
  whole gradient in one pass, but it returns the loss's type rather than the weights', so it
  cannot be asked for a tensor of weights
  ([vx-lang/Vx#1600](https://github.com/vx-lang/Vx/issues/1600)). So the gradient is taken one
  weight at a time: `loss_at(x, net, k)` is the loss with weight `k` set to `x`, and
  `grad(loss_at, weight(net, k), net, k)` is that weight's partial derivative. That is one pass
  per weight, nine here.
- The weights are a struct of `f64` fields, passed by value. Passing them as a pointer to a
  tensor breaks `grad` ([vx-lang/Vx#1601](https://github.com/vx-lang/Vx/issues/1601)).
- A program that uses `grad` runs only under `vxc -a run-jit`
  ([vx-lang/Vx#1594](https://github.com/vx-lang/Vx/issues/1594)), so `build.sh` runs it that way
  and there is no binary.

So this is a small network on purpose. `matmul` and tensors, which the plan for this project
named, wait on the first two issues.
