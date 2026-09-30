#!/usr/bin/env bash
# Read-only machine envelope for Apple Silicon LLM work. Never touches the GPU,
# so it is safe while an inference server holds it.
set -u
pg=$(pagesize 2>/dev/null || echo 16384)
gib() { awk -v b="$1" 'BEGIN{printf "%.1f", b/1073741824}'; }
vmv() { vm_stat | awk -v k="$1" -v pg="$pg" -F: '$1==k{gsub(/[ .]/,"",$2); print $2*pg}'; }

echo "arch:          $(uname -m)$( [ "$(sysctl -n sysctl.proc_translated 2>/dev/null)" = 1 ] && echo '  ROSETTA — run a native arm64 shell/python')"
echo "chip:          $(sysctl -n machdep.cpu.brand_string)  gpu_cores=$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/{print $2; exit}')"
echo "macos:         $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
mem=$(sysctl -n hw.memsize); echo "ram:           $(gib "$mem") GiB"
wl=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo n/a)
echo "wired_limit:   ${wl} MB$( [ "$wl" = 0 ] && echo ' (system default)')"
w=$(vmv 'Pages wired down'); an=$(vmv 'Anonymous pages'); c=$(vmv 'Pages occupied by compressor')
echo "wired_now:     $(gib "$w") GiB"
echo "anonymous:     $(gib "$an") GiB   compressed: $(gib "$c") GiB"
echo "others_gib:    $(gib $((w + an + c)))  (wired+anonymous+compressed; pass to fit.py --others-gib when no model is loaded)"
echo "free:          $(gib "$(vmv 'Pages free')") GiB"
echo "swap:          $(sysctl -n vm.swapusage)"
echo "swapouts:      $(vm_stat | awk -F: '/Swapouts/{gsub(/[ .]/,"",$2); print $2}') (cumulative; diff two samples)"
echo "pressure:      $(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null) (1 normal, 2 warn, 4 critical)"
echo "--- engines"
for b in llama-server llama-cli; do
  command -v $b >/dev/null && echo "$b: $($b --version 2>&1 | grep -m1 -i version)" || echo "$b: absent"
done
pys=$(command -v python3)
# conda is often lazy-loaded and absent from non-interactive PATH: also scan the standard roots
for root in ${CONDA_PREFIX:+"$CONDA_PREFIX"} ~/miniforge3 ~/mambaforge ~/miniconda3 ~/anaconda3 /opt/homebrew/Caskroom/miniforge/base; do
  [ -d "$root" ] && pys="$pys $root/bin/python $(ls -d "$root"/envs/*/bin/python 2>/dev/null)"
done
for py in $(printf "%s\n" $pys | awk '!seen[$0]++'); do
  [ -x "$py" ] || continue
  out=$("$py" -c 'import mlx_lm,mlx.core as m;print(mlx_lm.__version__,"mlx",m.__version__)' 2>/dev/null) && echo "mlx_lm [$py]: $out"
done
for b in ollama vllm lms; do command -v $b >/dev/null && echo "$b: $(command -v $b)" || echo "$b: absent"; done
echo "--- listeners (common LLM ports)"
for p in 8000 8080 1234 11434; do
  l=$(lsof -nP -iTCP:$p -sTCP:LISTEN 2>/dev/null | awk 'NR==2{print $1" pid="$2}')
  echo ":$p ${l:-free}"
done
