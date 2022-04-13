# !/bin/sh
cmake -B build
cmake --build build --config Release --parallel
cd build/src/
./inference  --use_cuda
