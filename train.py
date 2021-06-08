import argparse
import gym
import os
from utils.mpi_tools import mpi_fork
from algo.ppo.ppo import train
from env import create_train_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--exp_name', type=str, default='cnn')
    args = parser.parse_args()

    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    mpi_fork(args.cpu)  # run parallel code with mpi

    logger_kwargs = dict(
        output_dir=os.path.join(args.log_dir,args.exp_name,'seed'+str(args.seed)),
        exp_name=args.exp_name,
    )

    train(  lambda : create_train_env(1,1,'complex'),
            gamma=args.gamma, 
            seed=args.seed, 
            steps_per_epoch=args.steps,
            epochs=args.epochs,
            #max_ep_len=200,
            pi_lr=0.001,
            vf_lr=0.001,
            logger_kwargs=logger_kwargs)
