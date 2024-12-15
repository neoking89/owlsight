import os


class GGUF_Utils:
    """
    A utility class to determine optimal settings for GGUF models based on CPU count.
    Credits for the original implementation go to the authors of the easy-llama project.
    source: https://github.com/ddh0/easy-llama
    """

    _cpu_count = os.cpu_count()

    @classmethod
    def get_optimal_n_batch(cls) -> int:
        """
        Determines the optimal number of batches based on the CPU count.

        Returns:
            int: Optimal batch size.
        """
        cpu_count = cls._cpu_count
        # Use a power-of-2 scheme based on CPU count ranges
        if cpu_count <= 2:
            return 1
        elif cpu_count <= 4:
            return 2
        elif cpu_count <= 8:
            return 4
        elif cpu_count <= 16:
            return 8
        return 16

    @classmethod
    def get_optimal_n_threads(cls) -> int:
        """
        Determines the optimal number of threads based on the CPU count.

        Returns:
            int: Optimal thread count.
        """
        return max(cls._cpu_count // 2, 1)

    @classmethod
    def get_optimal_n_threads_batch(cls) -> int:
        """
        Returns the total number of available CPU cores.

        Returns:
            int: Total CPU core count.
        """
        return cls._cpu_count
