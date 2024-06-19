import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from envs.one_step_matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from torch.distributions import Categorical
from utils.th_utils import get_parameters_num
import math

class NQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))
        
        self.entropy_coef = 0.03

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params,  lr=args.lr)
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        
        self.train_t = 0
        
        self.n_agents = args.n_agents

        # th.autograd.set_detect_anomaly(True)
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1].to(self.device)
        actions = batch["actions"][:, :-1].to(self.device)
        terminated = batch["terminated"][:, :-1].float().to(self.device)
        mask = batch["filled"][:, :-1].float().to(self.device)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1]).to(self.device)
        avail_actions = batch["avail_actions"].to(self.device)
        
        dead_onehot = th.zeros_like(avail_actions[0,0,0])
        dead_onehot[0] = 1.
        dead_onehot = dead_onehot.int()
        dead_allies = avail_actions.clone() == dead_onehot
        dead_allies = dead_allies.all(-1).float()
        
        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        mac_out = self.mixer.func_g(mac_out, batch["state"], t_env)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time
            target_mac_out = self.target_mixer.func_g(target_mac_out, batch["state"], t_env)

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach = self.mixer.func_f(mac_out_detach, batch["state"], t_env)
            mac_out_detach = mac_out_detach / self.entropy_coef
            mac_out_detach[avail_actions == 0] = -9999999
            actions_pdf = th.softmax(mac_out_detach, dim=-1)
            rand_idx = th.rand(actions_pdf[:,:,:,:1].shape).to(actions_pdf.device)
            actions_cdf = th.cumsum(actions_pdf, -1)
            rand_idx = th.clamp(rand_idx, 1e-6, 1-1e-6)
            picked_actions = th.searchsorted(actions_cdf, rand_idx)
            target_qvals = th.gather(target_mac_out.clone(), 3, picked_actions).squeeze(3)
            
            target_logp = th.log(actions_pdf)
            target_logp = th.gather(target_logp, 3, picked_actions).squeeze(3)
            target_entropy = -target_logp.sum(-1, keepdim=True) 
            
            # logp_inf2zero = th.where(th.log(actions_pdf)==-th.inf, 0, th.log(actions_pdf))
            # target_entropy = -actions_pdf * logp_inf2zero
            # target_entropy = target_entropy.sum(-1).sum(-1, keepdim=True)
            
            # Calculate n-step Q-Learning targets
            target_qvals = self.target_mixer(target_qvals, batch["state"], target_mac_out)

            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                    self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_qvals, target_entropy*self.entropy_coef,
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer
        naive_sum = chosen_action_qvals.clone().detach().sum(-1, keepdim=True)
        chosen_aq_clone = chosen_action_qvals.clone().detach()
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], dead_allies[:,:-1])

        td_error = (chosen_action_qvals - targets.detach())
        td_error_sq = 0.5 * td_error.pow(2)
        mask = mask.expand_as(td_error_sq)
        masked_td_error = td_error_sq * mask
        L_td = masked_td_error.sum() / mask.sum()
        
        # beta loss
        affine_aq = self.mixer.func_f(chosen_aq_clone, batch["state"][:, :-1], t_env, dead_allies[:,:-1])
        approx_error = chosen_action_qvals.detach() - affine_aq.sum(-1, keepdim=True)
        beta_error = 0.5 * approx_error.pow(2)
        masked_beta_error = beta_error * mask
        L_beta = masked_beta_error.sum() / mask.sum()
        
        gopt_mask = (((approx_error>0.).float() + (td_error<0.).float()) != 1).float()
        weight_td_error = masked_td_error * gopt_mask * 0.5 + masked_td_error * (1-gopt_mask)
        mask_sum = mask * gopt_mask * 0.5 + mask * (1-gopt_mask)
        L_wtd = weight_td_error.sum() / mask_sum.sum()

        loss = L_wtd + L_beta

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("loss_wtd", L_wtd.item(), t_env)
            self.logger.log_stat("loss_beta", L_beta.item(), t_env)
            self.logger.log_stat("err_mask", (mask_sum.sum()/mask.sum()).item(), t_env)
            # self.logger.log_stat("loss_total", L_total.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("entropy", target_entropy.mean().item(), t_env)
            self.logger.log_stat("entropy_coef", self.entropy_coef, t_env)
            self.logger.log_stat("naive_sum", naive_sum.mean().item(), t_env)
            # self.logger.log_stat("beta_error", beta_error.mean().item(), t_env)
            self.log_stats_t = t_env
            
            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
