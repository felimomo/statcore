# statcore

A C++ package for some statistical fit functionalities.

Reproduce GLM model fits:

```
# Create + activate conda env
conda env create -f environment.yml 
conda activate statcore

# Configure + Compile
cmake -S . -B build -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run the benchmarks
./build/mod_bmks
python bench/mod_analysis.py 
```

Work in progress...