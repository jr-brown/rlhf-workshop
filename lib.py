
import jax
import math
import logging
import gymnasium as gym

import numpy as np
import jax.numpy as jnp
import jax.random as jrnd
import flax.linen as nn

from copy import copy
from tqdm import tqdm
from typing import Callable, Any, Optional, Generator, Iterable, Type, Sequence
from itertools import repeat
from functools import partial
from datetime import datetime
from dataclasses import dataclass

from jax.tree_util import tree_map

from jaxtyping import Array, Key, Shaped, Float, Bool

from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState

from gymnasium.vector.utils import batch_space
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.base_class import BaseAlgorithm

from gymnasium.vector.utils import batch_space

from pprint import pformat
from itertools import count

from optax import adamw

from flax.core import unfreeze
from flax.training.train_state import TrainState

from sbx import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.base_class import BaseAlgorithm

import reinforcement  # So that cartpole custom env is registered


log = logging.getLogger(__name__)


Kwargs = dict[str, Any]


@dataclass
class Trajectory:
    observations: Shaped[Array, "transitions *obs_features"]
    actions: Shaped[Array, "transitions *acts_features"]
    rewards: Optional[Float[Array, "transitions"]]
    infos: Optional[list[dict[str, Any]]]=None

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, val) -> "Trajectory":
        return Trajectory(
            self.observations[val],
            self.actions[val],
            None if self.rewards is None else self.rewards[val],
            None if self.infos is None else self.infos[val],
        )


class TrajectoryExtractorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._last_episode_starts = None
        self._ep_starts_buf = []
        self._obss_buf = []
        self._acts_buf = []
        self._rews_buf = []
        self._infos_buf = []

    def _on_training_start(self):
        pass

    def _on_rollout_start(self):
        pass

    def _on_step(self):
        if self._last_episode_starts is None:
            self._last_episode_starts = self.locals["self"]._last_episode_starts

        self._ep_starts_buf.append(self._last_episode_starts)
        self._obss_buf.append(self.locals["self"]._last_obs)
        self._acts_buf.append(self.locals["actions"])
        self._rews_buf.append(self.locals["rewards"])
        self._infos_buf.append(self.locals["infos"])

        self._last_episode_starts = self.locals["dones"]
        return True

    def _on_rollout_end(self):
        pass

    def _on_training_end(self):
        pass

    def empty(self) -> list[Trajectory]:
        if len(self._ep_starts_buf) < 1:
            return []

        [ep_starts, obss, acts, rews], [infos] = process_chunked_trajs(
            arrayable=[self._ep_starts_buf, self._obss_buf, self._acts_buf, self._rews_buf],
            non_arrayable=[self._infos_buf]
        )

        self._ep_starts_buf = []
        self._obss_buf = []
        self._acts_buf = []
        self._rews_buf = []
        self._infos_buf = []

        return trajData2trajs(RawTrajectoryData(ep_starts, obss, acts, rews, infos))


def venv2env_kwargs(vec_env_kwargs):
    env_kwargs = vec_env_kwargs.get("env_kwargs", {})
    env_kwargs["id"] = vec_env_kwargs["env_id"]
    return env_kwargs


def get_time_stamp(include_seconds: bool=False) -> str:
    date = datetime.now().date().__str__()
    time = datetime.now().time().__str__()

    if include_seconds:
        time = '-'.join(time.split(':')).split('.')[0]
    else:
        time = '-'.join(time.split(':')[:2])

    return f"{date}_{time}"


def process_chunked_trajs(arrayable: list, non_arrayable: list) -> tuple[list[Array], list[list]]:
    def process_non_arrayable(xs):
        new_xs = []
        for i in range(len(xs[0])):
            for x in xs:
                new_xs.append(x[i])
        return new_xs

    def process_arrayable(xs):
        return jnp.concatenate(xs.swapaxes(0, 1))

    return (
        [process_arrayable(jnp.array(xs)) for xs in tqdm(arrayable, desc="Processing arrayable traj chunk parts", leave=False)],
        [process_non_arrayable(xs) for xs in tqdm(non_arrayable, desc="Processing non-arrayable traj chunk parts", leave=False)]
    )


@dataclass
class RawTrajectoryData:
    episode_starts: Bool[Array, "transitions"]
    observations: Shaped[Array, "transitions *obs_features"]
    actions: Shaped[Array, "transitions *acts_features"]
    rewards: Optional[Float[Array, "transitions"]]
    infos: Optional[list[dict[str, Any]]]

    @classmethod
    def empty(cls) -> "RawTrajectoryData":
        return cls(jnp.empty(0), jnp.empty(0), jnp.empty(0), jnp.empty(0), [])

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, val) -> "RawTrajectoryData":
        return RawTrajectoryData(
            self.episode_starts[val],
            self.observations[val],
            self.actions[val],
            None if self.rewards is None else self.rewards[val],
            None if self.infos is None else self.infos[val],
        )

    def unpack(self) -> tuple[Array, Array, Array, Array | list[None],
                              list[dict[str, Any]] | list[None]]:
        if self.rewards is None:
            rewards = [None for _ in self.episode_starts]
        else:
            rewards = self.rewards

        if self.infos is None:
            infos = [None for _ in self.episode_starts]
        else:
            infos = self.infos

        assert len(self.episode_starts) == len(self.observations) == len(self.actions) == len(rewards) == len(infos)

        return self.episode_starts, self.observations, self.actions, rewards, infos


def trajData2trajs(traj_data: RawTrajectoryData) -> list[Trajectory]:
    ep_starts, obs, acts, rews, infos = traj_data.unpack()
    start_idxs = jnp.where(ep_starts == 1)[0]
    starts_ends = list(zip(start_idxs, start_idxs[1:])) + [(start_idxs[-1], len(ep_starts))]

    return [
        Trajectory(
            obs[start:end],
            acts[start:end],
            rews[start:end] if rews[start] is not None else None,
            infos[start:end] if infos[start] is not None else None,
        )
        for start, end in starts_ends
    ]


def venv_collect_trajectories(
    venv: VecEnv | Kwargs,
    timesteps: int,
    rng: Optional[Key]=None,
    agent: BaseAlgorithm | Kwargs | None =None,
) -> list[Trajectory]:

    if isinstance(venv, dict):
        venv_kwargs = replace_venv_str_with_cls(venv)
        seed = 0 if rng is None else int(jrnd.bits(rng))
        venv = make_vec_env(**venv_kwargs, seed=seed)

    if isinstance(agent, dict):
        log.info("Initialising agent")
        agent, = nones_to_empty_dicts(agent)
        agent = get_agent(venv, **agent)

    b_space = batch_space(venv.get_attr("action_space")[0], n=venv.num_envs)

    trajs = []

    obs = venv.reset()

    ep_starts = []
    obss = []
    acts = []
    rews = []
    infos = []

    prev_dones = None

    for _ in tqdm(range(timesteps // venv.num_envs), desc="Trajectory collect", leave=False):
        if agent is None:
            act = np.array(b_space.sample())
        else:
            act, _ = agent.predict(obs)

        new_obs, r, dones, info = venv.step(act)

        if prev_dones is None:
            ep_starts.append([True for _ in act])
        else:
            ep_starts.append(prev_dones)

        prev_dones = dones

        obss.append(obs)
        acts.append(act)
        rews.append(r)
        infos.append(info)

        obs = new_obs

    log.info("Processing trajectories")
    [ep_starts, obss, acts, rews], [infos] = process_chunked_trajs(
        arrayable=[ep_starts, obss, acts, rews],
        non_arrayable=[infos]
    )

    trajs = trajData2trajs(RawTrajectoryData(ep_starts, obss, acts, rews, infos))

    return trajs


def replace_venv_str_with_cls(vec_env_kwargs):
    vec_env_cls_table = {
        "subprocvecenv": SubprocVecEnv,
        "dummyvecenv": DummyVecEnv,
    }
    _vec_env_kwargs = copy(vec_env_kwargs)

    if "vec_env_cls" in vec_env_kwargs.keys():
        _vec_env_kwargs["vec_env_cls"] = vec_env_cls_table[vec_env_kwargs["vec_env_cls"].lower()]

    return _vec_env_kwargs


agent_algos: dict[str, Type[BaseAlgorithm]] = {
    "PPO": PPO,
    "SAC": SAC,
}


def get_agent(
    env,
    algorithm: str="PPO",
    policy_network: str="MlpPolicy",
    verbose=1,
    policy_kwargs: Optional[str]=None,
    name: str="",
    **kwargs,
) -> BaseAlgorithm:

    if name == "":
        name = get_time_stamp()

    if policy_kwargs is not None:
        # TODO: Restrict globals here
        policy_kwargs_d: Optional[dict[str, Any]] = eval(policy_kwargs)
    else:
        policy_kwargs_d = None

    return agent_algos[algorithm](
        policy_network,
        env,
        policy_kwargs=policy_kwargs_d,
        verbose=verbose,
        tensorboard_log=f".tensorboard_logs/{name}",
        **kwargs
    )


def cartpole(traj: Trajectory) -> float:
    return float(sum(jnp.cos(obs[2]) for obs in traj.observations))


def halfcheetah(
    traj: Trajectory,
    forward_reward_weight=1.0,
    ctrl_cost_weight=0.1,
) -> float:

    if traj.infos is not None:
        forward_reward = sum(i["reward_run"] for i in traj.infos)
        control_cost = sum(i["reward_ctrl"] for i in traj.infos)
    else:
        log.warn(f"Infos for halfcheetah trajectories not found, using fallback approximate method")
        forward_reward = forward_reward_weight * jnp.sum(traj.observations[::,8])
        control_cost = -ctrl_cost_weight * jnp.sum(jnp.square(traj.actions))

    return float(forward_reward + control_cost)


reward_fn_map: dict[str, Callable[[Trajectory], float]] = {
    "cartpole": cartpole,
    "halfcheetah": halfcheetah,
}


def get_thread_rng(rng_seed: int=0):
    return jrnd.fold_in(jrnd.PRNGKey(rng_seed), jax.process_index())


non_linearities: dict[str, Callable[[Array], Array]] = {
    "id": lambda xs: xs,
    "relu": nn.relu,
    "softmax": lambda xs: nn.softmax(xs, axis=-1),
    "softplus": nn.softplus,
    "sigmoid": nn.sigmoid,
}

normalisations: dict[str, Type[nn.Module]] = {
    "batch": NotImplemented,  # Need to pass around batch_stats in TrainState to use this
    "layer": nn.LayerNorm,
    "instance": nn.InstanceNorm,
}


class Ensemble(nn.Module):
    num_members: int
    base_model: nn.Module
    take_mean: bool=False

    @nn.compact
    def __call__(
        self,
        xs: Float[Array, "..."]
    ) -> Float[Array, "{self.num_members} ..."]:

        outs = jnp.array([self.base_model.copy()(xs) for _ in range(self.num_members)])
        return jnp.mean(outs, axis=0) if self.take_mean else outs


class MLP(nn.Module):
    layer_sizes: Sequence[int]
    flatten_dim_range: Optional[tuple[int, int]]=None  # Inclusive
    normalisation: Optional[str]=None
    normalisation_kwargs: Optional[dict]=None
    internal_non_linearities: str="relu"
    dropout_prob: float=0.0
    output_non_linearity: Optional[str]=None

    @nn.compact
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:

        assert x.size != 0
        assert len(self.layer_sizes) > 0

        if self.flatten_dim_range is not None:
            lower, raw_upper = self.flatten_dim_range
            upper = (raw_upper % len(x.shape)) + 1
            assert upper > lower
            x = jnp.reshape(x, (*x.shape[:lower], -1, *x.shape[upper:]))

        for n_features in self.layer_sizes[:-1]:
            x = nn.Dense(features=n_features)(x)

            if self.normalisation is not None:
                norm_kwargs = {} if self.normalisation_kwargs is None else self.normalisation_kwargs
                x = normalisations[self.normalisation](**norm_kwargs)(x)

            x = non_linearities[self.internal_non_linearities](x)
            x = nn.Dropout(rate=self.dropout_prob, deterministic=not self.has_rng("dropout"))(x)

        x = nn.Dense(features=self.layer_sizes[-1])(x)

        if self.output_non_linearity is not None:
            x = non_linearities[self.output_non_linearity](x)

        return x


def make_model(model_spec: str):
    eval_globals = {
        "nn": nn,
        "Ensemble": Ensemble,
        "MLP": MLP,
    }

    model = eval(model_spec, eval_globals)
    assert isinstance(model, nn.Module)

    return model


def filter_trim_in_iterable(xs: Iterable, desired_length: int) -> list:
    return [x[:desired_length] for x in xs if len(x) >= desired_length]


def partition_int(
    desired_sum: int,
    n_elems: Optional[int]=None,
    weights: Optional[list[float]]=None
) -> list[int]:

    if weights is not None:
        assert ((n_elems is not None) and (len(weights) == n_elems)) or (n_elems is None)
        normed_weights = [w/sum(weights) for w in weights]

    else:
        assert n_elems is not None
        normed_weights = [1/n_elems for _ in range(n_elems)]

    ideal = [w * desired_sum for w in normed_weights]
    current_pass = [math.floor(w) * desired_sum for w in normed_weights]

    while sum(current_pass) < desired_sum:
        diffs = [i - c for i, c in zip(ideal, current_pass)]
        current_pass[diffs.index(max(diffs))] += 1

    assert sum(current_pass) == desired_sum

    return current_pass


def nones_to_empty_dicts(*args):
    return [{} if d is None else d for d in args]


def shard(
    xs: Array | dict[str, Array],
    n_devices: Optional[int]=None
) -> dict[str, Array]:
    n = n_devices if n_devices is not None else jax.local_device_count()
    return jax.tree_map(lambda x: x.reshape((n, -1) + x.shape[1:]), xs)


class DeviceInfo:
    def __init__(self, ids: Optional[list[int]]=None, max_num: Optional[int]=None):

        if ids is not None:
            specific_devices = [x for x in jax.local_devices() if x.id in ids]
        else:
            specific_devices = None

        if (specific_devices is not None) and (specific_devices != []):
            devices = specific_devices
        else:
            devices = jax.devices()

        if max_num is not None:
            assert max_num > 0
            devices = devices[:max_num]

        log.info(f"Using {devices=}")
        self.devices = devices

    @property
    def num(self):
        return len(self.devices) if self.devices is not None else 1

    def unpack(self) -> tuple[Optional[list], int]:
        return self.devices, self.num


def get_pref_data(
    pref_fn: Callable[[list[Trajectory], list[Trajectory], Key], Array],
    num_prefs: int,
    fragment_length: int,
    total_trajs: list[Trajectory],
    recent_trajs: list[Trajectory],
    rng: Key,
    pref_max_old_traj_frac: float | None = None,  # If None samples all trajectories uniformly
) -> dict[str, Array]:

    if pref_max_old_traj_frac is None:
        trajs_for_pref = total_trajs

    else:
        assert pref_max_old_traj_frac < 1.0
        rng, traj_sample_rng = jrnd.split(rng)
        num_old_trajs = int(len(recent_trajs) * (pref_max_old_traj_frac / (1 - pref_max_old_traj_frac)))
        old_trajectory_idxs = jrnd.permutation(traj_sample_rng, len(total_trajs))[:num_old_trajs]
        trajs_for_pref = recent_trajs + [total_trajs[idx] for idx in old_trajectory_idxs]

    traj_maxlen = max([encode_traj(t).shape[0] for t in trajs_for_pref])
    assert fragment_length <= traj_maxlen, "Fragment length too large"

    log.info("Getting fragments for comparison")

    trajs_for_pref = [t for t in trajs_for_pref if encode_traj(t).shape[0] >= fragment_length]

    rng, order_rng, fragment_rng = jrnd.split(rng, num=3)

    chosen_idxs = jrnd.choice(order_rng, len(trajs_for_pref), (2*num_prefs,))
    fragment_starts = [jrnd.choice(jrnd.fold_in(fragment_rng, i),
                                   len(trajs_for_pref[idx]) - fragment_length)
                       for i, idx in enumerate(chosen_idxs)]

    fragments = [trajs_for_pref[idx][start:start+fragment_length]
                 for idx, start in zip(chosen_idxs, fragment_starts)]

    assert all([encode_traj(fragments[0]).shape == encode_traj(t).shape for t in fragments])

    fragments_a = fragments[:num_prefs]
    fragments_b = fragments[num_prefs:]

    prefs = pref_fn(fragments_a, fragments_b, rng)

    t0_xss = jnp.array([encode_traj(t) for t in fragments_a])
    t1_xss = jnp.array([encode_traj(t) for t in fragments_b])

    return {
        "prefs": prefs,
        "trajs_0": t0_xss,
        "trajs_1": t1_xss,
    }


def initialise(
    model_cls,
    gt_reward_fn: str | Callable[[Trajectory], float],
    traj_length: int,
    schedule_kwargs: Kwargs,
    venv_kwargs: Kwargs,
    reward_predictor_spec: str,
    rng: Key,

    agent_kwargs: Kwargs | None = None,
    agent_name: str="",
):
    random_rollout_steps, agent_ts_schedule, train_pref_schedule = get_schedules(**schedule_kwargs)

    schedule_iterator = tqdm(
        list(zip(count(), agent_ts_schedule, train_pref_schedule)),
        desc="Demo RLHF",
    )

    if isinstance(gt_reward_fn, str):
        gt_reward_fn = reward_fn_map[gt_reward_fn.lower()]

    log.info("Initialising vec env")
    rng, vec_env_seed_rng = jrnd.split(rng)
    venv_kwargs = replace_venv_str_with_cls(venv_kwargs)
    venv = make_vec_env(**venv_kwargs, seed=int(jrnd.bits(vec_env_seed_rng)))

    log.info("Initialising agent")
    agent_kwargs, = nones_to_empty_dicts(agent_kwargs)
    agent = get_agent(venv, name=agent_name, **agent_kwargs)

    log.info("Collecting random rollouts")
    rnd_trajs = venv_collect_trajectories(venv, random_rollout_steps, None)
    log.info(f"{len(rnd_trajs)=}")

    random_trajs = filter_trim_in_iterable(rnd_trajs, traj_length)
    log.info(f"{len(random_trajs)=}")

    log.info("Initialising reward model")

    model = model_cls(reward_net=make_model(reward_predictor_spec))
    log.info(f"{model=}")

    example_input = {
        "prefs": jnp.array([0]),
        "trajs_0": jnp.array([encode_traj(random_trajs[0])]),
        "trajs_1": jnp.array([encode_traj(random_trajs[1])]),
    }

    rng, init_rng = jrnd.split(rng)
    tx = adamw(learning_rate=0.001)
    params = model.init(init_rng, **example_input)['params']

    log.info(f"Initialised parameter shapes:\n{pformat(tree_map(jnp.shape, unfreeze(params)), width=100, compact=True)}")
    model_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    learnt_reward_fn = get_reward_fn(model, model_state.params)
    venv = VecEnvRewardFn(venv, learnt_reward_fn)
    agent.set_env(venv)

    return (
        agent,
        model,
        model_state,
        schedule_iterator,
        gt_reward_fn,
        random_trajs,
    )


def _format_action(act, dtype):
    act_arr = np.array(act, dtype=dtype)
    return dtype(act_arr.item()) if act_arr.size == 1 else act_arr


def visualise_policy(
    env_cfg: Kwargs,
    agent,
    steps_per_attempt: Optional[int]=None,
    attempts: int=1,
    log_total_rewards: bool=True,
):
    env = gym.make(**env_cfg, render_mode="human")
    dtype = np.int32 if isinstance(env.action_space, gym.spaces.Discrete) else np.float32

    for _ in range(attempts):
        total_reward = 0.0
        obs, _ = env.reset()

        attempt_iter = repeat(0) if steps_per_attempt is None else range(steps_per_attempt)
        for _ in attempt_iter:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(_format_action(action, dtype))
            total_reward += reward
            env.render()

            if terminated or truncated:
                break

        if log_total_rewards:
            log.info(total_reward)

    env.close()


def visualise_trajectories(
    env_cfg: Kwargs,
    trajectories: list[Trajectory],
    state_l2_tol: float | None =None,
):
    env = gym.make(**env_cfg, render_mode="human")
    dtype = np.int32 if isinstance(env.action_space, gym.spaces.Discrete) else np.float32

    for i, trajectory in enumerate(trajectories):
        log.info(f"Playing trajectory {i+1} of {len(trajectories)}")

        # Reset typically stochastic so fix on first observation and then monitor divergence
        _, _ = env.reset()
        env.unwrapped.state = trajectory.observations[0]
        actual_obs = trajectory.observations[0]

        for recorded_obs, action in zip(trajectory.observations, trajectory.actions):
            if (state_l2_tol is not None) and (jnp.linalg.norm(actual_obs - recorded_obs, ord=2) > state_l2_tol):
                # log.warning(f"States diverged, resetting")
                env.unwrapped.state = recorded_obs

            actual_obs, _, terminated, truncated, _ = env.step(_format_action(action, dtype))
            env.render()

            print(actual_obs)

            if terminated or truncated:
                break

    env.close()


def encode_traj(traj: Trajectory) -> Array:
    return jnp.concatenate([
        jnp.reshape(traj.observations, (len(traj.observations), -1)),
        jnp.reshape(traj.actions, (len(traj.actions), -1))
    ], axis=1)


def train_reward_model(
    model_state: TrainState,
    train_ds: dict[str, Array],
    batch_size: int,
    train_epochs: int,
    steps_per_eval: int | None,
    rng: Key,
    device_info: DeviceInfo,
) -> TrainState:

    devices, n_devices = device_info.unpack()

    assert batch_size % n_devices == 0

    rng, dropout_rng = jrnd.split(rng)

    p_train_step = jax.pmap(
        partial(train_step, rng=dropout_rng),
        axis_name="batch",
        devices=devices,
    )

    log.info("Getting steps for train_steps")
    ds_length = len(list(train_ds.values())[0])
    train_steps = math.ceil(train_epochs * ds_length / batch_size)

    steps_per_eval = 1 if steps_per_eval is None else steps_per_eval
    assert 1 <= steps_per_eval <= train_steps

    model_state = replicate(model_state, devices=devices)

    rng, perm_rng = jrnd.split(rng)
    batch_generator = get_batch_generator(train_ds, perm_rng, batch_size, n_devices=n_devices)

    batch_losses = []

    for step in tqdm(range(train_steps), desc="Train steps", leave=False):

        batch = next(batch_generator)
        model_state, loss = p_train_step(model_state, batch)

        batch_losses.append(loss)

        if step % steps_per_eval == 0:
            mean_loss = jnp.mean(jnp.array(batch_losses))
            batch_losses = []
            log.info(f"Train mean_loss={mean_loss:.5g}")

    return unreplicate(model_state)


def train_step(
    state: TrainState,
    batch: dict[str, Array],
    *,
    rng: Key,
    apply_fn_kwargs: dict[str, Any] | None=None,
) -> tuple[TrainState, Array]:

    apply_fn_kwargs, = nones_to_empty_dicts(apply_fn_kwargs)
    folded_dropout_rng = jrnd.fold_in(rng, state.step)

    def loss_fn(params: Array) -> tuple[Array, tuple[Array, dict | None]]:
        logits = state.apply_fn(
            {'params': params},
            **batch,
            rngs={"dropout": folded_dropout_rng},
        )

        loss = jnp.mean(logits)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, "batch")
    state = state.apply_gradients(grads=grads)

    return state, loss


def get_batch_generator(
    ds: dict[str, Array],
    rng: Key,
    batch_size: int,
    n_devices: Optional[int]=None,
) -> Generator[dict[str, Array], None, None]:

    length = len(list(ds.values())[0])

    def replenish_idx_buffers(
        _rng: Key,
        _idx_buffer: Array,
        _idx_batches_buffer: Array,
    ) -> tuple[Array, Array]:

        while len(_idx_buffer) < batch_size:
            _rng, perm_rng = jrnd.split(_rng)
            _idx_buffer = jnp.concatenate([_idx_buffer, jrnd.permutation(perm_rng, length)])

        batches = len(_idx_buffer) // batch_size
        _idx_batches_buffer = iter(_idx_buffer[:batches * batch_size].reshape((batches, batch_size)))
        _idx_buffer = _idx_buffer[batches * batch_size:]

        return _idx_buffer, _idx_batches_buffer

    rng, perm_rng = jrnd.split(rng)
    idx_buffer = jrnd.permutation(perm_rng, length)
    idx_batches_buffer = iter(jnp.array([]))

    while True:
        try:
            idxs = next(idx_batches_buffer)
            yield tree_map(lambda x: shard(x[idxs, ...], n_devices=n_devices), ds)

        except StopIteration:
            rng, replenish_rng = jrnd.split(rng)
            idx_buffer, idx_batches_buffer = replenish_idx_buffers(
                replenish_rng, idx_buffer, idx_batches_buffer
            )


def get_reward_fn(module: nn.Module, params: dict | nn.FrozenDict) -> Callable:
    return jax.jit(partial(
        module.reward_net.apply,
        {"params": params["reward_net"]},
    ))


def get_schedules(
    rlhf_iters: int,
    env_timestep_budget: int,
    pref_comparison_budget: int=0,
    random_rollouts_relative_weight: float=1,
    initial_comparisons_importance: float=1,
) -> tuple[int, list[int], list[int]]:

    random_rollout_steps, *agent_ts_schedule = partition_int(
        env_timestep_budget,
        weights=[random_rollouts_relative_weight] + list(repeat(1.0, rlhf_iters))
    )

    train_pref_schedule = partition_int(
        pref_comparison_budget,
        weights=[initial_comparisons_importance] + list(repeat(1.0, rlhf_iters-1))
    )

    log.info(f"\n{random_rollout_steps=}\n{agent_ts_schedule=}\n{train_pref_schedule=}")

    return random_rollout_steps, agent_ts_schedule, train_pref_schedule


class VecEnvRewardFn(VecEnvWrapper):
    def __init__(
        self,
        venv: VecEnv,
        reward_fn: Callable,
    ):
        assert not isinstance(venv, VecEnvRewardFn)
        super().__init__(venv)
        self.reward_fn = reward_fn
        self._last_obss = None
        self._acts = None
        self.reset()

    @property
    def envs(self):
        return self.venv.envs

    def reset(self):
        self._last_obss = self.venv.reset()
        return self._last_obss

    def step_async(self, actions):
        self._acts = actions
        return self.venv.step_async(actions)

    @partial(jax.jit, static_argnums=(0))
    def get_new_rewards(self, obss, last_obss, acts):

        acts = jnp.array(acts)

        if acts.ndim == last_obss.ndim - 1:
            acts = jnp.reshape(acts, (*acts.shape, -1))
        else:
            assert acts.ndim == last_obss.ndim

        xs = jnp.concatenate([last_obss, acts], axis=-1)

        rews = self.reward_fn(xs)
        assert len(rews) == len(obss), f"{rews.shape} {obss.shape}"

        # Reshape as otherwise it can have a trailing 1-length dim
        return rews.reshape(*obss.shape[:-1])


    def step_wait(self):
        obss, _, dones, infos = self.venv.step_wait()

        assert self._last_obss is not None
        assert self._acts is not None

        rews = self.get_new_rewards(
            jnp.array(obss),
            jnp.array(self._last_obss),
            jnp.array(self._acts),
        )

        rews = np.array(rews)
        self._last_obss = obss

        return obss, rews, dones, infos


def update_reward_fn(
    model: BaseAlgorithm,
    reward_fn: Callable,
):
    def set_reward_fn(venv):
        if isinstance(venv, VecEnvRewardFn):
            venv.reward_fn = reward_fn

        else:
            venv.env = set_reward_fn(venv.env)

        return venv

    venv = model.get_env()
    venv = set_reward_fn(venv)
    model.set_env(venv)

    return model

