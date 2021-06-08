from mpi4py import MPI
import os, subprocess, sys
import numpy as np

# 退出当前脚本主进程，并在MPI环境中启动n个当前脚本进程
def mpi_fork(n, bind_to_core=False):
    """
    Re-launches the current script with workers linked by MPI.

    Also, terminates the original process that launched it.

    Taken almost without modification from the Baselines function of the
    `same name`_.

    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py

    Args:
        n (int): Number of process to split into.

        bind_to_core (bool): Bind each MPI process to a core.
    """
    if n<=1: 
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        # for mpich
        #args = ["mpiexec", "-np", str(n)]
        # for openmpi
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


def msg(m, string=''):
    print(('Message from %d: %s \t '%(MPI.COMM_WORLD.Get_rank(), string))+str(m))

def proc_id():
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank()

# 全约归，不再有root的概念，每个进程的recvbuf都是一样操作、一样结果的
def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)

def num_procs():
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size()

# 默认用0号进程来统一
def broadcast(x, root=0):
    MPI.COMM_WORLD.Bcast(x, root=root)

# 在allreduce的基础上封装的操作的基本函数，主要是为了统一标量和序列操作。
def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff

def mpi_sum(x):
    return mpi_op(x, MPI.SUM)

def mpi_avg(x):
    """Average a scalar or vector over MPI processes."""
    return mpi_sum(x) / num_procs()
    
# 获取进程所有数组的mean和std
def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std