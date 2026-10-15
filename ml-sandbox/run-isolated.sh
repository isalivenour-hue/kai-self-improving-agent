#!/usr/bin/env bash
set -euo pipefail
root="$HOME/ml-sandbox"
script="${1:?usage: run-isolated.sh experiments/script.py}"
case "$script" in experiments/*.py) ;; *) echo 'Only scripts under experiments/ are allowed.' >&2; exit 2;; esac
exec /usr/bin/bwrap \
  --die-with-parent --unshare-pid \
  --ro-bind /usr /usr --ro-bind /bin /bin --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
  --ro-bind "$root/etc" /etc --ro-bind "$HOME/ml-basics-user" "$HOME/ml-basics-user" \
  --bind "$root" "$root" --tmpfs /tmp --proc /proc --dev /dev \
  --setenv HOME "$root/home" --setenv MPLCONFIGDIR "$root/home/.matplotlib" \
  --chdir "$root" "$HOME/ml-basics-user/bin/python" "$root/$script"
