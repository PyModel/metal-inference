"""Mac benchmark guard. Host usage, GPU allocation and process footprint have different scopes."""
import ctypes
import fcntl
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

from fit import GiB, gpu_alloc, vm_bytes


def output(*args):
    return subprocess.check_output(args, text=True, timeout=5)


@contextmanager
def gpu_lock():
    path = Path(tempfile.gettempdir()) / f"metal-inference-{os.getuid()}.lock"
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another benchmark holds the GPU lock") from None
        yield


def power():
    source = output("pmset", "-g", "batt").splitlines()[0]
    profile = re.findall(r"^\s*((?:lowpower|highpower|power)mode)\s+(\d+)", output("pmset", "-g"), re.M)
    return {"source": source, "profile": dict(profile)}


def model_servers():
    # Match process arguments at token boundaries; handle python -m mlx_lm.server.
    rows = output("ps", "-axo", "pid=,command=").splitlines()
    servers = {}
    mlx_commands = {"mlx_lm.server", "mlx_lm.generate", "mlx_lm.benchmark"}
    native_commands = {"llama-server", "llama-bench", "llama-cli", "llama-batched-bench",
                       "llama-perplexity", "llama-completion", "ds4", "ds4-server"}
    for row in rows:
        pid, command = row.strip().split(maxsplit=1)
        args = command.split()
        name = os.path.basename(args[0])
        if name.lower().startswith("python"):
            script = args[2:3] if args[1:2] == ["-m"] else args[1:2]
            if script and os.path.basename(script[0]) in mlx_commands:
                name = os.path.basename(script[0])
        if name in native_commands | mlx_commands or (name == "ollama" and "runner" in args[1:3]):
            servers[int(pid)] = name
    return servers


def footprint(pid):
    # Darwin rusage_info_v0: UUID, then ten 64-bit fields; ri_phys_footprint is field 8.
    data = (ctypes.c_uint64 * 12)()
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    if lib.proc_pid_rusage(pid, 0, ctypes.byref(data)) != 0:
        raise OSError(ctypes.get_errno(), "cannot read server physical footprint")
    return data[9]


def sample(pid=None):
    vm = output("vm_stat")
    _, used = vm_bytes(vm)
    gpu = gpu_alloc()
    if gpu is None:
        raise RuntimeError("GPU allocation probe unavailable")
    swap = re.search(r"^Swapouts:\s*(\d+)", vm, re.M)
    return {"host_used_gib": used / GiB, "gpu_alloc_gib": gpu / GiB,
            "swapouts": int(swap[1]), "pressure": int(output("sysctl", "-n", "kern.memorystatus_vm_pressure_level")),
            **({"server_footprint_gib": footprint(pid) / GiB} if pid else {})}


def check(sample, initial, caps):
    for key, limit in caps.items():
        if sample[key] > limit:
            raise RuntimeError(f"{key} above cap: {sample[key]:.2f} > {limit:.2f} GiB")
    if sample["pressure"] != 1 or sample["swapouts"] > initial["swapouts"]:
        raise RuntimeError("memory pressure or new swapouts")
