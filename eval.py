import time
import joblib
import os
import os.path as osp
import numpy as np
import torch
from utils.logx import EpochLogger
from env import create_train_env
from algo.ppo.core import MarioActor

device = torch.device('cuda')
def load_pytorch_policy(fpath, itr='', deterministic=False):

    fname = osp.join(fpath, 'pyt_save', 'model' + itr + '.pt')
    print('\n\nLoading from %s.\n\n' % fname)

    # model = MarioActor(4,12,torch.nn.ReLU).cuda()
    # model.load_state_dict(torch.load(fname))
    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            # pi, _ = model(x.to(device), None).cpu()
            # action = pi.sample()
            prob_logit = model.logits_net(x.to(device)).cpu()
            action = np.argmax(prob_logit)
        return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger('.')

    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-2*3)

        a = get_action(o)
        o, r, d, _ = env.step(a.numpy().item())
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()
    os.remove('./progress.txt')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', '-f', type=str, default='log/cnn/w1s1')
    parser.add_argument('--len', '-l', type=int, default=5000)
    parser.add_argument('--episodes', '-n', type=int, default=1)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()

    env = create_train_env(1,1, 'complx')
    # from gym import wrappers
    # env = wrappers.Monitor(env,"./video/", force=True)
    get_action = load_pytorch_policy(args.fpath)#itr='_50'
    run_policy(env, get_action, args.len, args.episodes, not (args.norender))
    env.close()
