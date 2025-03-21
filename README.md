# Goldbach Comet Project

This project visualizes the Goldbach conjecture through innovative data analysis and computational techniques. It produces stunning comet-like visualizations of Goldbach partition counts for even numbers.

## Features

- High-performance GPU-accelerated calculations using CUDA/cupy
- Multi-threaded CPU fallback processing
- Dynamic visualization with configurable parameters 
- Support for large number ranges (10M+)
- Optimized rendering using OpenGL acceleration
- Progress tracking and performance metrics
- Multiple optimization approaches across versions

## Performance 

Test results on RTX 5080:

- 100k numbers: ~17.32s computation, ~3.70s rendering
- 1M numbers: ~32.49s computation, ~6.41s rendering 
- 10M numbers: ~483.13s computation, ~59.14s rendering
- Peak GPU usage: 70-80%, ~195W power draw, ~60°C

## Requirements

- Python 3.8+
- CUDA toolkit 
- Dependencies:
    - numpy
    - cupy
    - matplotlib 
    - mplcairo
    - tqdm
    - numba

## Usage

```bash
python G_x_v1.5.py
```

Key parameters can be configured at the top of the script:
- MAX_N: Maximum number to analyze
- SAMPLE_SIZE: Number of points to sample
- divvs: Divisors for color patterns
- DPI: Output image resolution

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

MIT License