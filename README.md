# ML experiment sandbox

Experiments run with:
- a private writable directory: `~/ml-sandbox`
- the Python environment mounted read-only
- a temporary `/tmp`
- full network access enabled (DNS verified)
- no access to the rest of the home directory

Run `~/ml-sandbox/run-isolated.sh experiments/real_ai_from_scratch.py`.
