import torch.nn as nn
import os
import torch
import numpy as np
import pickle
from torch.nn import functional as F
import torchvision.transforms as transforms

try:
    from detr.main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
except:
    from .detr.main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
import IPython

e = IPython.embed


class ACTPolicy(nn.Module):

    def __init__(self, args_override, RoboTwin_Config=None):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override, RoboTwin_Config)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        print(f"KL Weight {self.kl_weight}")

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict["mse"] = mse
            loss_dict["loss"] = loss_dict["mse"]
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


# ---------------------------------------------------------------------------
# Inference-mode selection per task (success-rate lever; NO architecture/ckpt
# change -- same state_dict, only when/how predicted chunks get executed).
#
# Official T2-T4 eval runs THIS package's code (the submitted --code-dir shadows
# policy/ACT via sys.path) but reads deploy_policy.yml + CLI flags from the
# organizer's own runtime, where temporal_agg is always forced False. So this
# dict is the ONE switch guaranteed to reach official evaluation. A/B locally
# first (starter/eval_local.py + the env vars below), then pin the winner here
# before submitting.
#
# Grounding: the ACT paper's ablations (arXiv:2304.13705) show chunking drives
# success (1%->44% going k=1->100) but full open-loop execution loses
# reactivity, and temporal ensembling recovers ~+3.3% by smoothing
# chunk-boundary jumps; BID (arXiv:2408.17355, ICLR'25) confirms the
# chunk-boundary consistency/reactivity tradeoff is a first-order failure
# source. Supported knobs per task:
#   "temporal_agg": True    per-step re-query + exp-weighted ensemble (k=0.01)
#   "query_frequency": N    partial-chunk execution: re-query every N < chunk
#                           steps, run only the first N actions of each chunk
# Local A/B env overrides (highest precedence; no code edit needed):
#   ACT_TEMPORAL_AGG=0|1  ACT_QUERY_FREQUENCY=N
TASK_INFERENCE_OVERRIDES = {
    # grab_roller: quasi-static two-arm lift, 400-step limit -> open-loop chunk
    # 50 gives only 8 corrections/episode; ensemble smooths the synchronized
    # lift. Pinned per the paper's ablation -- CONFIRM with eval_local A/B.
    "grab_roller": {"temporal_agg": True},
}


def _resolve_inference_mode(args_override):
    """(temporal_agg, query_frequency_override) with precedence:
    env (local A/B) > TASK_INFERENCE_OVERRIDES (submitted default) > caller args."""
    ov = dict(TASK_INFERENCE_OVERRIDES.get(args_override.get("task_name") or "", {}))
    env_ta = os.environ.get("ACT_TEMPORAL_AGG")
    env_qf = os.environ.get("ACT_QUERY_FREQUENCY")
    if env_ta is not None or env_qf is not None:
        ov = {}  # an env-driven A/B run fully defines the mode; code defaults step aside
        if env_ta is not None:
            ov["temporal_agg"] = env_ta == "1"
        if env_qf is not None:
            ov["query_frequency"] = int(env_qf)
    temporal_agg = ov.get("temporal_agg", bool(args_override.get("temporal_agg", False)))
    query_frequency = ov.get("query_frequency") or args_override.get("query_frequency")
    return temporal_agg, query_frequency


class ACT:

    def __init__(self, args_override=None, RoboTwin_Config=None):
        if args_override is None:
            args_override = {
                "kl_weight": 0.1,  # Default value, can be overridden
                "device": "cuda:0",
            }
        self.policy = ACTPolicy(args_override, RoboTwin_Config)
        self.device = torch.device(args_override["device"])
        self.policy.to(self.device)
        self.policy.eval()

        # Temporal aggregation settings
        self.temporal_agg, _qf_override = _resolve_inference_mode(args_override)
        self.num_queries = args_override["chunk_size"]
        self.state_dim = RoboTwin_Config.action_dim  # Standard joint dimension for bimanual robot
        self.max_timesteps = 3000  # Large enough for deployment

        # Set query frequency based on temporal_agg - matching imitate_episodes.py logic
        self.query_frequency = self.num_queries
        if _qf_override:
            self.query_frequency = max(1, min(int(_qf_override), self.num_queries))
        print(f"[ACT] task={args_override.get('task_name')!r} inference mode: "
              f"temporal_agg={self.temporal_agg} "
              f"query_frequency={1 if self.temporal_agg else self.query_frequency}/{self.num_queries}")
        if self.temporal_agg:
            self.query_frequency = 1
            # Initialize with zeros matching imitate_episodes.py format
            self.all_time_actions = torch.zeros([
                self.max_timesteps,
                self.max_timesteps + self.num_queries,
                self.state_dim,
            ]).to(self.device)
            print(f"Temporal aggregation enabled with {self.num_queries} queries")

        self.t = 0  # Current timestep

        # Load statistics for normalization
        ckpt_dir = args_override.get("ckpt_dir", "")
        if ckpt_dir:
            # Load dataset stats for normalization
            stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
            if os.path.exists(stats_path):
                with open(stats_path, "rb") as f:
                    self.stats = pickle.load(f)
                print(f"Loaded normalization stats from {stats_path}")
            else:
                raise FileNotFoundError(
                    f"dataset_stats.pkl not found at {stats_path} -- refusing to eval without "
                    f"normalization (would yield a meaningless success rate)."
                )

            # Load policy weights
            ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
            print("current pwd:", os.getcwd())
            if os.path.exists(ckpt_path):
                loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
                print(f"Loaded policy weights from {ckpt_path}")
                print(f"Loading status: {loading_status}")
            else:
                raise FileNotFoundError(
                    f"policy checkpoint not found at {ckpt_path} -- refusing to eval a randomly "
                    f"initialised model (would yield a meaningless success rate)."
                )
        else:
            self.stats = None

    def pre_process(self, qpos):
        """Normalize input joint positions"""
        if self.stats is not None:
            return (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        return qpos

    def post_process(self, action):
        """Denormalize model outputs"""
        if self.stats is not None:
            return action * self.stats["action_std"] + self.stats["action_mean"]
        return action

    def get_action(self, obs=None):
        if obs is None:
            return None

        # Convert observations to tensors and normalize qpos - matching imitate_episodes.py
        qpos_numpy = np.array(obs["qpos"])
        qpos_normalized = self.pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos_normalized).float().to(self.device).unsqueeze(0)

        # Prepare images following imitate_episodes.py pattern
        # Stack images from all cameras. Order MUST match training: process_data writes
        # cam_high, cam_right_wrist, cam_left_wrist (head, RIGHT, LEFT). Stacking head,left,right
        # swaps the two wrist views and the model consumes cameras positionally -> silently wrong
        # eval inputs (codex High).
        curr_images = []
        camera_names = ["head_cam", "right_cam", "left_cam"]
        for cam_name in camera_names:
            curr_images.append(obs[cam_name])
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            # Only query the policy at specified intervals - exactly like imitate_episodes.py
            if self.t % self.query_frequency == 0:
                self.all_actions = self.policy(qpos, curr_image)

            if self.temporal_agg:
                # Match temporal aggregation exactly from imitate_episodes.py
                self.all_time_actions[[self.t], self.t:self.t + self.num_queries] = (self.all_actions)
                actions_for_curr_step = self.all_time_actions[:, self.t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]

                # Use same weighting factor as in imitate_episodes.py
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = (torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1))

                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                # Direct action selection, same as imitate_episodes.py
                raw_action = self.all_actions[:, self.t % self.query_frequency]

        # Denormalize action
        raw_action = raw_action.cpu().numpy()
        action = self.post_process(raw_action)

        self.t += 1
        return action