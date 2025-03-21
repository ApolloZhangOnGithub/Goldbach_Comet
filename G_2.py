from typing import List, Dict
from functools import lru_cache

@lru_cache(maxsize=1000)
def is_prime(n: int) -> bool:
    """
    Check if a number is prime using cache for better performance.
    
    Args:
        n (int): Number to check
        
    Returns:
        bool: True if prime, False otherwise
    """
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def prime_factorization_vector(n: int) -> List[int]:
    """
    Compute the prime factorization vector of a number.
    
    Args:
        n (int): Number to factorize
        
    Returns:
        List[int]: Vector containing prime factor counts
        
    Raises:
        ValueError: If input is less than 1
        TypeError: If input is not an integer
    """
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")
    if n < 1:
        raise ValueError("Input must be a positive integer")

    # Find prime factors and their counts
    factors: Dict[int, int] = {}
    i = 2
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i
        i += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1

    # Create vector of prime factor counts
    max_prime = max(factors.keys()) if factors else 2
    result = []
    for prime in range(2, max_prime + 1):
        if is_prime(prime):
            result.append(factors.get(prime, 0))
    
    return result

if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Generate data for first 100 numbers
        numbers = list(range(1, 101))
        vectors = [prime_factorization_vector(n) for n in numbers]
        
        # Convert vectors to numpy array with padding
        max_len = max(len(v) for v in vectors)
        padded_vectors = [v + [0] * (max_len - len(v)) for v in vectors]
        data = np.array(padded_vectors)

        # Create heatmap
        plt.figure(figsize=(15, 8))
        plt.imshow(data.T, aspect='auto', cmap='YlOrRd')
        plt.colorbar(label='Exponent count')
        plt.xlabel('Number')
        plt.ylabel('Prime index')
        plt.title('Prime Factorization Vectors (1-100)')
        
        # Add prime numbers as y-axis labels - fixed version
        primes = [p for p in range(2, max_len + 2) if is_prime(p)]
        positions = list(range(len(primes)))  # Create matching positions for the primes
        plt.yticks(positions, primes)
        
        plt.show()

    except (ValueError, TypeError) as e:
        print(f"Error: {e}")
    except ImportError:
        print("Please install matplotlib and numpy to visualize the results")