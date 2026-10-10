cmake -S . -B build -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build build && ./build/scratch
cmake --build build && ./build/sum_bmk