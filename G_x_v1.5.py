#v1.1 更新内容：添加进度条
#v1.2 更新内容：优化运算，利用多核CPU多线程加速，1m计算量-4质数-16核32线程约10min(Ryzen 9950x测试数据)
#v1.3 更新内容：进一步优化运算，如果可行则利用GPU计算单元加速
#   v1.3下测试：100k计算量-4质数，在CPU运算下约7.59sec运算和1.94sec绘图(Ryzen 9950x-16核32线程 测试数据))
#   v1.3下测试：500k计算量-4质数，在CPU运算下约166.30sec运算和6.41sec绘图(Ryzen 9950x-16核32线程 测试数据))
#   v1.3下测试：100k计算量-4质数，在GPU运算下约112.42sec运算和1.92sec绘图(RTX5080 测试数据)
#   v1.3下测试：500k计算量-4质数，在GPU运算下约898.94sec运算和6.68sec绘图(RTX5080 测试数据)
#v1.4 更新内容：优化计算逻辑，采用CUDA加速质数检查，采用GPU埃拉托斯特尼筛法预生成质数表，采用双指针法计算哥德巴赫组合数，显著加速GPU版本
#   v1.4下测试：100k计算量-4质数，在GPU运算下约17.32sec运算和3.70sec绘图(RTX5080 测试数据)
#   v1.4下测试：1m计算量-4质数，在GPU运算下约32.49sec运算和6.41sec绘图(RTX5080 测试数据)
#   v1.4下测试：10m计算量-4质数，在GPU运算下约483.13sec运算和59.14sec绘图，瞬时功耗月195W，GPU温度约60度，GPU占用在70%-80%浮动(RTX5080 测试数据)
#v1.5 更新内容：采用对数均匀降采样、颜色计算向量化、渲染优化、内存优化、质数筛法改进、双指针法 GPU 优化改进性能，但注意结果绘图使用了采样，仅展示趋势，并非完整点集

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Value, Manager
from functools import partial
import time
import cupy as cp
from numba import cuda
import matplotlib
matplotlib.use('Agg')  # 非交互式后端（最快）
# matplotlib.use('WebAgg')  # 网页交互式后端（次优）
import mplcairo
matplotlib.use("module://mplcairo.base")  # OpenGL 加速

# Global GPU flag
USE_GPU = True
MAX_N = 100000 + 1
SAMPLE_SIZE = MAX_N//2
STEP = MAX_N//SAMPLE_SIZE
divvs = [2,3,5,7,9,11,13,15]
DPI=1200
CT = False

@cuda.jit
def is_prime_kernel(n_array, results):
    idx = cuda.grid(1)
    if idx >= n_array.size:
        return
    n = n_array[idx]
    if n < 2:
        results[idx] = False
        return
    max_div = int(n ** 0.5) + 1
    for d in range(2, max_div):
        if n % d == 0:
            results[idx] = False
            return
    results[idx] = True

def is_prime_gpu(n_array):
    if not HAS_GPU:
        return np.array([is_prime(n) for n in n_array])
    
    n_gpu = cp.asarray(n_array)
    results_gpu = cp.zeros(n_array.size, dtype=cp.bool_)
    
    threads_per_block = 256
    blocks_per_grid = (n_array.size + threads_per_block - 1) // threads_per_block
    
    is_prime_kernel[blocks_per_grid, threads_per_block](n_gpu, results_gpu)
    return results_gpu.get()

def generate_primes_gpu(max_n):
    sieve = cp.ones(max_n + 1, dtype=cp.bool_)
    sieve[0:2] = False
    sieve[2::2] = False  # 标记偶数
    sieve[2] = True
    
    # 分段筛法优化显存占用
    sqrt_n = int(cp.sqrt(max_n))
    for i in range(3, sqrt_n + 1, 2):
        if sieve[i]:
            sieve[i*i::i] = False
    return cp.where(sieve)[0]

def initialize_gpu():
    """Initialize GPU and return GPU status"""
    global HAS_GPU
    if not USE_GPU:
        print("GPU 模式已禁用")
        return False

    print("GPU 模式已启用")
    try:
        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        cuda_path = os.environ.get('CUDA_PATH', 'Not set')
        
        if cuda_path == 'Not set':
            print("警告：CUDA_PATH 环境变量未设置")
            raise RuntimeError("CUDA_PATH not set")
            
        required_dlls = ['nvrtc64_120_0.dll', 'cudart64_12.dll']
        missing_dlls = []
        for dll in required_dlls:
            dll_path = os.path.join(cuda_path, 'bin', dll)
            if not os.path.exists(dll_path):
                missing_dlls.append(dll)
        
        if missing_dlls:
            raise RuntimeError(f"缺少必要的 CUDA DLL 文件: {', '.join(missing_dlls)}")
        
        HAS_GPU = True
        return True
    except Exception as e:
        print(f"GPU 初始化失败: {e}")
        print("切换到 CPU 模式")
        return False

memo: dict[int, bool] = {}
def is_prime(n: int) -> bool:
    if n in memo:
        return memo[n]
    if n < 2:
        memo[n] = False
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            memo[n] = False
            return False
    memo[n] = True
    return True

def is_prime_gpu(n_array):
    if not HAS_GPU:
        return np.array([is_prime(n) for n in n_array])
    
    n = cp.asarray(n_array)
    result = (n >= 2)  # 初始化为 True 当且仅当 n >= 2
    
    # 生成所有可能的除数 (2 到 sqrt(max(n)))
    max_sqrt = int(cp.sqrt(cp.max(n))) + 1
    divisors = cp.arange(2, max_sqrt)
    
    # 向量化检查所有除数 (利用广播机制)
    # 生成一个 2D 掩码矩阵：mask[i, j] = (n[i] % divisors[j] == 0) & (n[i] != divisors[j])
    mask = (n[:, None] % divisors == 0) & (n[:, None] != divisors)
    
    # 如果任何除数能整除，则标记为非质数
    result &= ~cp.any(mask, axis=1)
    
    return cp.asnumpy(result)
    

def count_goldbach_combinations(n):
    if n < 4 or n % 2 != 0:
        return 0
    count = 0
    for i in range(2, n//2 + 1):
        if is_prime(i) and is_prime(n-i):
            count += 1
    return count

def count_goldbach_combinations_gpu(n_array):
    if not HAS_GPU:
        return np.array([count_goldbach_combinations(n) for n in n_array])
    
    primes = generate_primes_gpu(cp.max(n_array))
    primes_device = primes  # 质数表已在 GPU
    
    n = cp.asarray(n_array)
    counts = cp.zeros_like(n, dtype=cp.int32)
    
    # 仅处理偶数且 >=4
    even_mask = (n >= 4) & (n % 2 == 0)
    even_n = n[even_mask]
    if even_n.size == 0:
        return cp.asnumpy(counts)
    
    # 双指针法向量化计算
    for p in primes_device:
        q = even_n - p
        # 利用二分查找快速判断 q 是否在质数表中
        q_in_primes = cp.isin(q, primes_device, assume_unique=True)
        counts[even_mask] += q_in_primes
    
    # 每个组合被计算了两次 (p, q) 和 (q, p)，需除以 2
    counts[even_mask] = counts[even_mask] // 2
    return cp.asnumpy(counts)

def process_chunk(use_gpu, numbers):
    """Process a chunk of numbers with GPU status awareness"""
    return [count_goldbach_combinations(i) for i in numbers]

# Generate data using parallel processing
def parallel_goldbach_count(start, end, step):
    """Generate data using parallel processing"""
    x = np.array(range(start, end, step))
    global HAS_GPU
    
    if HAS_GPU:
        try:
            print("Using GPU acceleration")
            y = count_goldbach_combinations_gpu(x)
            return x.tolist(), y.tolist()
        except Exception as e:
            print(f"GPU computation failed, falling back to CPU: {e}")
            HAS_GPU = False
    
    print("Using CPU computation")
    num_elements = len(x)
    available_cores = cpu_count()
    optimal_cores = min(available_cores, num_elements)
    chunk_size = max(1, num_elements // optimal_cores)
    chunks = []
    
    for i in range(0, num_elements, chunk_size):
        chunk = x[i:i + chunk_size]
        if len(chunk) > 0:
            chunks.append(chunk.tolist())
    
    # Create a partial function with GPU status
    process_chunk_with_gpu = partial(process_chunk, HAS_GPU)
    
    with Pool(processes=len(chunks)) as pool:
        results = list(tqdm(
            pool.imap(process_chunk_with_gpu, chunks),
            total=len(chunks),
            desc=f"Processing on {len(chunks)-1} threads"
        ))
    
    flattened = []
    for result in results:
        flattened.extend(result)
    
    return x.tolist(), flattened

# Generate all Goldbach pairs
def generate_pairs(random_sample, sample_size):
    pairs = []
    numbers = np.arange(4, MAX_N+1, 1)
    
    if random_sample:
        # Create logarithmic weights for numbers from 4 to MAX_N
        weights = np.log(numbers)
        weights = weights / np.sum(weights)  # Normalize weights
        # Sample numbers according to logarithmic distribution
        selected_numbers = np.random.choice(numbers, size=sample_size, p=weights)
    else:
        # Use all numbers
        selected_numbers = numbers
    
    for n in selected_numbers:
        for i in range(2, n//2 + 1):
            if is_prime(i) and is_prime(n-i):
                pairs.append((n, i))
    return pairs

def prepare_plot_data(args):
    idx, DIVV, x_data, y_data = args
    points = []
    colors = []
    for x_i, y_i in zip(x_data, y_data):
        points.append((x_i, y_i))
        colors.append("red" if x_i%DIVV==0 else ((1/DIVV)*(x_i%DIVV), 
                                          (1/DIVV)*(x_i%DIVV), 
                                          (1.0/DIVV)*((DIVV-x_i%DIVV)%DIVV)))
    return idx, DIVV, points, colors

def plot_processed_data(main_fig, plot_data, num_plots):
    idx, DIVV, points, colors = plot_data
    ax = main_fig.add_subplot(((num_plots + 1) // 2), 2, idx + 1)
    
    # Plot points efficiently using scatter
    x_coords, y_coords = zip(*points)
    ax.scatter(x_coords, y_coords, 
              s=40000/MAX_N,  # size parameter for scatter
              c=colors,
              marker='.')
    
    ax.set_title(f'DIVV={DIVV}: Number of Goldbach Combinations\nfor Even Numbers < {MAX_N}')
    ax.set_xlabel('Even Number')
    ax.set_ylabel('Number of Combinations')
    ax.grid(True)
    return ax

if __name__ == '__main__':
    print("Program started")
    print(f"CPU cores available: {cpu_count()}")
    
    # Initialize GPU only once in the main process
    HAS_GPU = initialize_gpu()
    print(f"GPU available: {HAS_GPU}")
    
    # Start computation timing
    compute_start = time.time()
    
    print(f"\nProcessing range: 4 to {MAX_N} with step {STEP}")
    x, y = parallel_goldbach_count(4, MAX_N, STEP)
    print(f"Processed {len(x)} numbers")
    
    compute_time = time.time() - compute_start
    print(f"\nComputation time: {compute_time:.2f} seconds ({compute_time/60:.2f} minutes)")
    
    # Start plotting timing
    plot_start = time.time()
    
    print("\nPreparing plots...")
    # Setup for parallel plotting
    num_plots = len(divvs)
    plot_args = [(idx, DIVV, x, y) for idx, DIVV in enumerate(divvs)]
    
    # Create main figure
    main_fig = plt.figure(figsize=(20, 4 * ((num_plots + 1) // 2)))
    
    # Process plot data in parallel
    with Pool(processes=min(cpu_count(), len(divvs))) as pool:
        processed_data = list(tqdm(
            pool.imap_unordered(prepare_plot_data, plot_args),
            total=len(divvs),
            desc="Processing plot data"
        ))
    
    # Sort processed data by index
    processed_data.sort(key=lambda x: x[0])
    
    # Create plots from processed data
    for plot_data in processed_data:
        plot_processed_data(main_fig, plot_data, num_plots)
    
    print("\nGenerating plot...")
    plt.tight_layout()
    
    # Calculate plot time before showing the plot
    plot_time = time.time() - plot_start
    print(f"\nPlotting time: {plot_time:.2f} seconds ({plot_time/60:.2f} minutes)")
    
    total_time = compute_time + plot_time
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Show plot after timing calculations
    #plt.show()
    plt.savefig(f'./output/goldbach_{MAX_N-1}_div={divvs}_dpi={DPI}_{time.strftime("%Y%m%d_%H%M%S")}.png', dpi=DPI, bbox_inches='tight')
    print("Plot saved to goldbach.png")