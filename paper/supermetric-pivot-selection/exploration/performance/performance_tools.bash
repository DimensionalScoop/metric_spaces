perf_perf(){
  perf record --freq=1000 --call-graph=fp .venv/bin/python -X perf $1
}

perf_lines(){
  mkdir -p logs
  LINE_PROFILE=1 uv run $1
  uv run python -m line_profiler -rtmz profile_output.lprof | moar -lang py -no-linenumbers -style base16-snazzy
  mv profile_output* logs/
}
