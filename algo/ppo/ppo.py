import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from env import create_train_env
import algo.ppo.core as core
from algo.ppo.buffer import PPOBuffer
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(env_fn, actor=core.MarioActor,critic=core.MarioCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=20, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n

    # Create actor-critic module
    ac_pi = actor(obs_dim[0], act_dim, activation=torch.nn.ReLU)
    ac_v = critic(obs_dim[0], activation=torch.nn.ReLU)

    # Sync params across processes
    sync_params(ac_pi)
    sync_params(ac_v)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac_pi, ac_v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, env.action_space.shape, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac_pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac_v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac_pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac_v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac_pi)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac_pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac_v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        # logger.store(LossPi=pi_l_old, LossV=v_l_old,
        #              KL=kl, Entropy=ent, ClipFrac=cf,
        #              DeltaLossPi=(loss_pi.item() - pi_l_old),
        #              DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            with torch.no_grad():
                x = torch.as_tensor(o, dtype=torch.float32).to(device)
                pi, _ = ac_pi(x)
                a = pi.sample() 
                logp = pi.log_prob(a)
                v = ac_v(x)

            next_o, r, d, _ = env.step(a.cpu().numpy().item())
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a.cpu().numpy(), r, v.cpu().numpy(), logp.cpu().numpy())

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d #or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    with torch.no_grad():
                        v =ac_v(torch.as_tensor(o, dtype=torch.float32).to(device)).cpu().numpy()
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0


        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.run_pytorch_saver()

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        # logger.log_tabular('LossPi', average_only=True)
        # logger.log_tabular('LossV', average_only=True)
        # logger.log_tabular('DeltaLossPi', average_only=True)
        # logger.log_tabular('DeltaLossV', average_only=True)
        # logger.log_tabular('Entropy', average_only=True)
        # logger.log_tabular('KL', average_only=True)
        # logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()