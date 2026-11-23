#!/usr/bin/env python3
"""Measure synchronized GPU copy traffic and FP16 GEMM with the installed MLX Python."""
import argparse
import statistics
import time

from mac_guard import gpu_lock, model_servers, power, sample


def median_seconds(op, mx):
    mx.eval(op())
    elapsed = []
    for _ in range(7):
        start = time.perf_counter()
        mx.eval(op())
        elapsed.append(time.perf_counter() - start)
    return statistics.median(elapsed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", type=int, default=4096)
    p.add_argument("--copy-mib", type=int, default=256)
    args = p.parse_args()
    if not 256 <= args.matrix <= 8192 or not 16 <= args.copy_mib <= 512:
        p.error("choose matrix size 256..8192 and copy MiB 16..512")
    with gpu_lock():
        state = power()
        initial = sample()
        if model_servers() or initial["pressure"] != 1 or "AC Power" not in state["source"]:
            p.exit(1, "requires an idle model server host, AC and normal pressure\n")
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        count = args.copy_mib * 2**20 // 4
        x = mx.random.uniform(shape=(count,))
        copy = mx.fast.metal_kernel(name="measured_copy", input_names=["x"], output_names=["y"],
                                    source="uint i = thread_position_in_grid.x; y[i] = x[i];")

        def copy_once():
            return copy(inputs=[x], grid=(count, 1, 1), threadgroup=(256, 1, 1),
                        output_shapes=[x.shape], output_dtypes=[x.dtype])[0]

        mx.eval(x)
        assert mx.array_equal(x, copy_once()).item()
        copy_s = median_seconds(copy_once, mx)
        n = args.matrix
        a, b = [mx.random.uniform(shape=(n, n), dtype=mx.float16) for _ in range(2)]
        mx.eval(a, b)
        gemm_s = median_seconds(lambda: a @ b, mx)
        final = sample()
        if power() != state or final["pressure"] != 1 or final["swapouts"] != initial["swapouts"] or model_servers():
            p.exit(1, "host state changed; discard calibration\n")
        print(f"copy read+write GB/s: {8 * count / copy_s / 1e9:.2f} ({args.copy_mib} MiB one way)")
        print(f"FP16 GEMM TFLOP/s: {2 * n**3 / gemm_s / 1e12:.2f} ({n} cubed)")


if __name__ == "__main__":
    main()
