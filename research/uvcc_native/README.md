## `uvcc_native` (PARALLEL.txt native core)

This directory contains a new, self-contained native (C++) core that incrementally implements
`research/PARALLEL.txt` phases, while keeping the existing Python UVCC runners intact.

### Build (local)

```bash
cmake -S research/uvcc_native -B research/uvcc_native/build -DCMAKE_BUILD_TYPE=Release
cmake --build research/uvcc_native/build -j
ctest --test-dir research/uvcc_native/build --output-on-failure
```


