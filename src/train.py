import os
import argparse
import torch
import numpy as np
from tensordict import TensorDict
import pickle
from scipy.optimize import linear_sum_assignment
from Codebook_Learning_RL.DataPrep import dataPrep
from Codebook_Learning_RL.env_ddpg import envCB
from Codebook_Learning_RL.DDPG_classes import OUNoise, GaussianNoise
from Codebook_Learning_RL.clustering import KMeans_only
from Codebook_Learning_RL.function_lib import bf_gain_cal, corr_mining
from save_results import MetricsWriter, save_run_data, plot_run
import time
import re
import copy
import random
import yaml
from tqdm.auto import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import scipy.io as scio
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
from modularl.policies.gaussian_policy import GaussianPolicy
from modularl.q_functions import SAQNetwork
from modularl.replay_buffers import ReplayBuffer
import torch.optim as optim
from create_agent import CreateAgent


SCENARIO_PATHS = {
    "los": "src/Codebook_Learning_RL/grid1101-1400.mat",
    "nlos": "src/Codebook_Learning_RL/grid201-300.mat",
}


class Config(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


def make_dirs_and_open(file_path, mode):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return open(file_path, mode)


def sanitize_filename(name):
    return re.sub(r"[^\w\-_\. ]", "_", name)


def getdatetime():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def parse_args():
    parser = argparse.ArgumentParser(description="Beam codebook RL training")
    parser.add_argument("--algo", type=str, default=None,
                        help="RL algorithm: sac / td3 / ddpg")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scenario", type=str, default=None,
                        choices=list(SCENARIO_PATHS.keys()),
                        help="Channel scenario: los / nlos")
    parser.add_argument("--num_NNs", type=int, default=None,
                        help="Codebook size (number of beams / RL agents)")
    parser.add_argument("--num_ant", type=int, default=None,
                        help="Number of BS antennas")
    parser.add_argument("--num_loop", type=int, default=None,
                        help="Number of outer training loops")
    parser.add_argument("--num_iter", type=int, default=None,
                        help="RL update steps per outer loop")
    parser.add_argument("--reward_type", type=str, default=None,
                        choices=["binary", "ternary"],
                        help="binary: {-1,+1}  ternary: {-1,0,+1} (paper Eq.22)")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    # ── Quantum VQC settings (used when algo=quantum_sac) ─────────────────
    parser.add_argument("--n_qubits", type=int, default=None,
                        help="VQC qubits. Defaults to num_ant (1:1 mapping).")
    parser.add_argument("--n_layers", type=int, default=None,
                        help="VQC variational depth (number of RY+RZ+CNOT blocks).")
    parser.add_argument("--diff_method", type=str, default=None,
                        choices=["backprop", "parameter-shift"],
                        help="PennyLane gradient method. "
                             "backprop=fast sim-only; parameter-shift=hardware-compatible.")
    return parser.parse_args()


def train(env, options, train_options, agent, beam_id, writer, noise):

    device = train_options["device"]
    CB_Env = env
    if train_options["overall_iter"] == 1:
        state = torch.zeros((1, options["num_ant"])).float().to(device)
    else:
        state = train_options["state"]

    iteration = 0
    num_of_iter = train_options["num_iter"]
    while iteration < num_of_iter:

        action = agent.act_train(state)
        reward_pred, _, action_quant_pred, state_1_pred = (
            CB_Env.get_reward(action.to(device))
        )
        reward_pred = torch.from_numpy(reward_pred).float().to(device)

        algo = train_options["algo"]
        if algo == "sac":
            # SAC handles exploration through stochastic policy internally
            action_pred_noisy = action
        else:
            # TD3: Gaussian noise (paper Eq. 11); DDPG: Ornstein-Uhlenbeck noise
            action_pred_noisy = noise.get_action(
                action, t=train_options["overall_iter"]
            )

        mat_dist = torch.abs(
            action_pred_noisy.reshape(options["num_ant"], 1)
            - options["ph_table_rep"]
        )
        action_quant = options["ph_table_rep"][
            range(options["num_ant"]), torch.argmin(mat_dist, dim=1)
        ].reshape(1, -1)

        state_1, reward, _, terminal = CB_Env.step(action_quant)
        reward = torch.from_numpy(reward).float().to(device)
        action = action_quant.reshape((1, -1)).float().to(device)

        batch_transition = TensorDict(
            {
                "observations": state.detach().clone(),
                "next_observations": state_1.detach().clone(),
                "actions": action.detach().clone(),
                "rewards": reward.detach().clone(),
                "dones": torch.tensor([terminal] * state.shape[0]),
            },
            batch_size=[state.shape[0]],
        )
        agent.rb.extend(batch_transition)

        agent.observe(
            state.detach().clone(),
            action_quant_pred.detach().clone(),
            reward_pred.detach().clone(),
            state_1_pred.detach().clone(),
            torch.tensor([terminal] * state.shape[0]),
        )

        iteration += 1
        train_options["overall_iter"] += 1
        state = state_1

        new_gain = (
            torch.Tensor.cpu(CB_Env.achievement)
            .detach()
            .numpy()
            .reshape((1, 1))
        )
        max_previous_gain = max(CB_Env.gain_history)
        if new_gain > max_previous_gain:
            CB_Env.gain_history.append(float(new_gain[0][0]))
        else:
            CB_Env.gain_history.append(float(max_previous_gain))

    train_options["state"] = state
    train_options["best_state"] = CB_Env.best_bf_vec
    if writer is not None:
        writer.add_scalars(
            f"gain_records/Beamforming_gain_beam_{beam_id}",
            {
                "gain": float(max_previous_gain),
                "EGC": float(CB_Env.compute_EGC()),
            },
            agent.global_step,
        )

    return train_options


def main():
    args = parse_args()

    experiments_dir = os.path.join("experiments", "runs")
    env_config_path = os.path.join("experiments", "configs", "env_config.yaml")
    algo_config_path = os.path.join("experiments", "configs", "algo_config.yaml")

    with open(algo_config_path) as f:
        train_opt = yaml.load(f, Loader=yaml.FullLoader)
    with open(env_config_path) as f:
        options = yaml.load(f, Loader=yaml.FullLoader)

    # ── Apply CLI overrides ───────────────────────────────────────────────
    if args.algo:                train_opt["algo"]        = args.algo
    if args.seed is not None:    train_opt["seed"]        = args.seed
    if args.reward_type:         train_opt["reward_type"] = args.reward_type
    if args.exp_name:            train_opt["exp_name"]    = args.exp_name
    if args.num_iter is not None: train_opt["num_iter"]   = args.num_iter
    if args.device:              train_opt["device"]      = args.device
    # Quantum VQC overrides
    if args.n_qubits is not None: train_opt["n_qubits"]   = args.n_qubits
    if args.n_layers is not None: train_opt["n_layers"]   = args.n_layers
    if args.diff_method:          train_opt["diff_method"] = args.diff_method

    if args.scenario:            options["path"]    = SCENARIO_PATHS[args.scenario]
    if args.num_NNs is not None: options["num_NNs"] = args.num_NNs
    if args.num_ant is not None: options["num_ant"] = args.num_ant
    if args.num_loop is not None: options["num_loop"] = args.num_loop
    if args.device:              options["device"]  = args.device

    # reward_type lives in train_opt but env also needs it
    options["reward_type"] = train_opt.get("reward_type", "binary")

    # ── Derived training settings ─────────────────────────────────────────
    train_opt["high_action"]  = torch.pi
    train_opt["low_action"]   = -torch.pi
    train_opt["action_shape"] = options["num_ant"]
    train_opt["obs_shape"]    = options["num_ant"]

    # n_qubits defaults to num_ant → direct 1:1 qubit-antenna mapping
    if not train_opt.get("n_qubits"):
        train_opt["n_qubits"] = options["num_ant"]

    algo_config = Config(train_opt)
    env_config  = Config(options)

    exp_name = sanitize_filename(algo_config.exp_name)
    run_name = f"{exp_name}__{getdatetime()}"[:255]
    run_dir  = os.path.join(experiments_dir, run_name)

    main_tb_writer = SummaryWriter(run_dir)
    writer = MetricsWriter(main_tb_writer)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join([f"|{k}|{v}|" for k, v in vars(algo_config).items()]),
    )
    writer.add_text(
        "env_parameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join([f"|{k}|{v}|" for k, v in vars(env_config).items()]),
    )

    create_agent = CreateAgent()
    random.seed(algo_config.seed)
    np.random.seed(algo_config.seed)
    torch.manual_seed(algo_config.seed)
    torch.backends.cudnn.deterministic = algo_config.torch_deterministic
    torch.autograd.set_detect_anomaly(True)

    with make_dirs_and_open(os.path.join(run_dir, "configs/algo_config.yaml"), "w") as f:
        yaml.dump(train_opt, f, indent=4, default_flow_style=False)
    with make_dirs_and_open(os.path.join(run_dir, "configs/env_config.yaml"), "w") as f:
        yaml.dump(options, f, indent=4, default_flow_style=False)

    os.makedirs(os.path.join(run_dir, "beams"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "beamforming_gain_records"), exist_ok=True)

    ch = dataPrep(options["path"])
    ch = np.concatenate(
        (
            ch[:, : options["num_ant"]],
            ch[:, int(ch.shape[1] / 2) : int(ch.shape[1] / 2) + options["num_ant"]],
        ),
        axis=1,
    )
    if options["path"] == "src/Codebook_Learning_RL/grid201-300.mat":
        ch = ch[2500:17500, :]
        print(f"NLoS scenario — {ch.shape[0]} users, {options['num_ant']} antennas")
    else:
        print(f"LoS scenario  — {ch.shape[0]} users, {options['num_ant']} antennas")

    with torch.cuda.device(options["device"]):
        u_classifier, sensing_beam = KMeans_only(
            ch, options["num_NNs"], n_bit=options["num_bits"], n_rand_beam=30
        )
        np.save(os.path.join(run_dir, "sensing_beam.npy"), sensing_beam)
        sensing_beam = torch.from_numpy(sensing_beam).float().to(options["device"])

        pickle.dump(u_classifier, open(os.path.join(run_dir, "kmeans_model.sav"), "wb"))

        options["num_ph"] = 2 ** options["num_bits"]
        options["multi_step"] = (
            torch.from_numpy(
                np.linspace(
                    int(-(options["num_ph"] - 2) / 2),
                    int(options["num_ph"] / 2),
                    num=options["num_ph"],
                    endpoint=True,
                )
            )
            .float().reshape(1, -1).to(options["device"])
        )
        options["pi"] = torch.tensor(np.pi).to(options["device"])
        options["ph_table"] = (2 * options["pi"]) / options["num_ph"] * options["multi_step"]
        options["ph_table"].to(options["device"])
        options["ph_table_rep"] = options["ph_table"].repeat(options["num_ant"], 1)

    env_list        = []
    train_opt_list  = []
    agent_list      = []
    noise_list      = []
    agent_writers   = []
    ou_noise_records = []   # only used for DDPG

    for beam_id in range(options["num_NNs"]):
        train_opt_list.append(copy.deepcopy(train_opt))

        agent_tb_writer = SummaryWriter(os.path.join(run_dir, f"NN_results_{beam_id}"))
        agent_writer = MetricsWriter(agent_tb_writer)
        agent_writers.append(agent_writer)
        ou_noise_records.append([])

        env_list.append(
            envCB(
                ch,
                options["num_ant"],
                options["num_bits"],
                beam_id,
                options,
                run_dir,
                train_opt["device"],
            )
        )

        if train_opt["algo"] in ("sac", "quantum_sac"):
            if train_opt["algo"] == "sac":
                agent = create_agent.sac(train_opt, agent_writer)
            else:
                agent = create_agent.quantum_sac(train_opt, agent_writer)
            noise_list.append(None)  # SAC family: stochastic policy handles exploration
        elif train_opt["algo"] == "td3":
            agent = create_agent.td3(train_opt, agent_writer)
            # TD3 uses Gaussian noise (paper Eq. 11), NOT Ornstein-Uhlenbeck
            noise_list.append(
                GaussianNoise(
                    (1, options["num_ant"]),
                    train_opt["device"],
                    max_sigma=train_opt.get("td3_noise_max", 0.5),
                    min_sigma=train_opt.get("td3_noise_min", 0.05),
                    decay_period=train_opt.get("td3_noise_decay", 100000),
                )
            )
        elif train_opt["algo"] == "ddpg":
            agent = create_agent.ddpg(train_opt, agent_writer)
            ou = OUNoise((1, options["num_ant"]), train_opt["device"])
            noise_list.append(ou)
            ou_noise_records[beam_id].append(ou.state.cpu().numpy().tolist())
        else:
            raise ValueError(f"Unknown algo: {train_opt['algo']}")

        agent_list.append(agent)

    # ── Main training loop ────────────────────────────────────────────────
    with torch.cuda.device(options["device"]):
        for _ in tqdm(range(options["num_loop"])):

            # Sampling
            n_sample = int(ch.shape[0] * options["ch_sample_ratio"])
            ch_sample_id = np.random.permutation(ch.shape[0])[:n_sample]
            ch_sample = (
                torch.from_numpy(ch[ch_sample_id, :]).float().to(options["device"])
            )

            # Clustering
            bf_mat_sample = bf_gain_cal(sensing_beam, ch_sample)
            f_matrix = corr_mining(bf_mat_sample, options["device"])
            f_matrix_np = torch.Tensor.cpu(f_matrix).numpy()
            labels = u_classifier.predict(np.transpose(f_matrix_np).astype(float))

            user_group, ch_group = [], []
            for ii in range(options["num_NNs"]):
                idx = np.where(labels == ii)[0].tolist()
                user_group.append(idx)
                ch_group.append(ch_sample[idx, :])

            # Assignment (Hungarian algorithm)
            best_beam_mtx = (
                torch.zeros((options["num_NNs"], 2 * options["num_ant"]))
                .float().to(options["device"])
            )
            for pp in range(options["num_NNs"]):
                best_beam_mtx[pp, :] = env_list[pp].best_bf_vec
            gain_mtx = bf_gain_cal(best_beam_mtx, ch_sample)

            cost_cols = [
                torch.mean(gain_mtx[:, user_group[ii]], dim=1).reshape(options["num_NNs"], -1)
                for ii in range(options["num_NNs"])
            ]
            cost_mtx = -torch.Tensor.cpu(torch.cat(cost_cols, dim=1)).numpy()
            row_ind, col_ind = linear_sum_assignment(cost_mtx)
            assignment = dict(zip(row_ind.tolist(), col_ind.tolist()))

            for ii in range(options["num_NNs"]):
                env_list[ii].ch = ch_group[assignment[ii]].to(train_opt_list[ii]["device"])
                env_list[ii].EGC_history.append(env_list[ii].compute_EGC())

            # RL update
            for beam_id in range(options["num_NNs"]):
                train_opt_list[beam_id] = train(
                    env_list[beam_id],
                    options,
                    train_opt_list[beam_id],
                    agent_list[beam_id],
                    beam_id,
                    writer,
                    noise_list[beam_id],
                )
                if train_opt["algo"] == "ddpg":
                    ou_noise_records[beam_id].append(
                        noise_list[beam_id].state.cpu().numpy().tolist()
                    )

    writer.close()

    # ── Save all artefacts ────────────────────────────────────────────────
    num_beam = options["num_NNs"]
    num_ant  = options["num_ant"]

    # Gain records (legacy per-beam npy under beamforming_gain_records/)
    for beam_id in range(num_beam):
        egc = env_list[beam_id].compute_EGC()
        gain_record = env_list[beam_id].gain_history[1:]
        gain_record.append(egc)
        gain_record = np.array(gain_record) / num_ant
        np.save(
            os.path.join(run_dir, "beamforming_gain_records", f"beam_{beam_id}_gain_records"),
            gain_record,
        )

    # Beam codebook (fixed: no longer overwritten by noise_records)
    raw = np.empty((num_beam, 2 * num_ant))
    for beam_id in range(num_beam):
        beam_path = os.path.join(run_dir, "beams", f"beams_{beam_id}_max.txt")
        if os.path.exists(beam_path):
            with open(beam_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            raw[beam_id, :] = np.fromstring(lines[-1], sep=",")

    beam_codebook = (1 / np.sqrt(num_ant)) * (raw[:, ::2] + 1j * raw[:, 1::2])
    np.save(os.path.join(run_dir, "beams", "beam_codebook.npy"), beam_codebook)
    scio.savemat(
        os.path.join(run_dir, "beams", "beam_codebook.mat"), {"beams": beam_codebook}
    )

    # OUNoise state history (DDPG only)
    if train_opt["algo"] == "ddpg" and any(len(r) for r in ou_noise_records):
        np.save(
            os.path.join(run_dir, "ou_noise_records.npy"),
            np.array(ou_noise_records, dtype=object),
            allow_pickle=True,
        )

    # Comprehensive results: raw npy + JSON metrics + figures
    all_gains, all_egc, all_metrics = save_run_data(
        run_dir, env_list, agent_writers, options, train_opt
    )
    plot_run(run_dir, all_gains, all_egc, all_metrics, options, train_opt)

    print(f"\nRun complete. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
