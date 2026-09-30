#!/usr/bin/env bash
# Read-only machine envelope for Apple Silicon LLM work. Never touches the GPU,
# so it is safe while a server (ds4, llama-server, mlx_lm.server) holds it.
set -u
pg=16384  # vm_stat pages are 16 KiB on Apple Silicon
gib() { awk -v b="$1" 'BEGIN{printf "%.1f", b/1073741824}'; }
vmv() { vm_stat | awk -v k="$1" -F: '$1==k{gsub(/[ .]/,"",$2); print $2*'$pg'}'; }

echo "arch:          $(uname -m)$( [ "$(sysctl -n sysctl.proc_translated 2>/dev/null)" = 1 ] && echo '  ROSETTA — native arm64 required')"
echo "chip:          $(sysctl -n machdep.cpu.brand_string)  gpu_cores=$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/{print $2; exit}')"
echo "macos:         $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
mem=$(sysctl -n hw.memsize); echo "ram:           $(gib "$mem") GiB"
wl=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
echo "wired_limit:   ${wl} MB$( [ "$wl" = 0 ] && echo ' (default; resets on reboot)')"
echo "wired_now:     $(gib "$(vmv 'Pages wired down')") GiB"
echo "anonymous:     $(gib "$(vmv 'Anonymous pages')") GiB   compressed: $(gib "$(vmv 'Pages occupied by compressor')") GiB"
echo "free:          $(gib "$(vmv 'Pages free')") GiB"
echo "swap:          $(sysctl -n vm.swapusage)"
echo "swapouts:      $(vm_stat | awk -F: '/Swapouts/{gsub(/[ .]/,"",$2); print $2}') (cumulative; diff two samples)"
echo "pressure:      $(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null) (1 normal, 2 warn, 4 critical)"
echo "--- engines"
v() { command -v "$1" >/dev/null && echo "$1: $("$@" 2>&1 | grep -m1 -i version)" || echo "$1: absent"; }
v llama-server --version
for e in mlx mlx-native; do
  py=~/miniforge3/envs/$e/bin/python
  [ -x "$py" ] && echo "mlx_lm[$e]: $("$py" -c 'import mlx_lm,mlx.core as m;print(mlx_lm.__version__,"mlx",m.__version__)' 2>&1 | tail -1)"
done
for b in ollama vllm; do command -v $b >/dev/null && echo "$b: $(command -v $b)" || echo "$b: absent"; done
echo "--- listeners"
for p in 8000 8080 8081 11434; do
  l=$(lsof -nP -iTCP:$p -sTCP:LISTEN 2>/dev/null | awk 'NR==2{print $1" pid="$2}')
  echo ":$p ${l:-free}"
done
