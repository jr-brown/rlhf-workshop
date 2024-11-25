
import jax
import logging

import jax.numpy as jnp
import jax.random as jrnd
import flax.linen as nn

from copy import copy
from functools import partial

from jax.nn import log_softmax, one_hot

from jaxtyping import Array, Float, Key

from flax.training.train_state import TrainState

from stable_baselines3.common.base_class import BaseAlgorithm

from lib import(
    visualise_trajectories,
    initialise,
    get_reward_fn,
    update_reward_fn,
    train_reward_model,
    get_pref_data,
    visualise_policy,
    Trajectory,
    nones_to_empty_dicts,
    filter_trim_in_iterable,
    Kwargs,
    DeviceInfo,
    get_thread_rng,
    venv2env_kwargs,
    TrajectoryExtractorCallback,
)


log = logging.getLogger(__name__)


class PreferenceModel(nn.Module):
    reward_net: nn.Module

    @nn.compact
    def __call__(
        self,
        prefs: Float[Array, "*batch"],
        traj_1: Float[Array, "*batch transitions features"],
        traj_2: Float[Array, "*batch transitions features"],
    ) -> Float[Array, ""]:

        if prefs.size == 0:
            return jnp.array([0])

        # Rewards per step of the trajectories
        traj_1_rewards = self.reward_net(traj_1).squeeze(-1)
        traj_2_rewards = self.reward_net(traj_2).squeeze(-1)

        # Total reward for each Trajectory
        r1 = jnp.sum(traj_1_rewards, axis=-1)
        r2 = jnp.sum(traj_2_rewards, axis=-1)

        # TODO: IMPLEMENT LOSS FUNCTION HERE
        raise NotImplementedError


def oracle_preferences(
    gt_reward_fn,
    fragments_a: list[Trajectory],
    fragments_b: list[Trajectory],
    rng: Key,
) -> Array:

    t1_rewards  = jnp.array([gt_reward_fn(t) for t in fragments_a])
    t2_rewards  = jnp.array([gt_reward_fn(t) for t in fragments_b])
    probs = jax.nn.sigmoid((t2_rewards - t1_rewards).astype(jnp.float32))
    return jrnd.bernoulli(rng, probs).astype(float)


def rlhf(
    traj_length: int,
    pref_fragment_length: int,
    initialise_kwargs: Kwargs,
    venv_kwargs: Kwargs,
    rm_train_kwargs: Kwargs | None = None,
    agent_train_kwargs: Kwargs | None = None,
    initial_rm_train_multiplier: int | None = None,
    pref_max_old_traj_frac: float | None = None,  # If None samples all trajectories uniformly
    rng_seed: int=0,
    device_info_kwargs: Kwargs | None = None,
    num_to_visualise_per_iteration: int | None = None,
) -> tuple[BaseAlgorithm, tuple[PreferenceModel, TrainState], list[Trajectory]]:

    (
        rm_train_kwargs, agent_train_kwargs, device_info_kwargs
    )= nones_to_empty_dicts(
        rm_train_kwargs, agent_train_kwargs, device_info_kwargs
    )

    device_info = DeviceInfo(**device_info_kwargs)

    rng = get_thread_rng(rng_seed=rng_seed)
    rng, ini_rng = jrnd.split(rng)

    (
        agent,
        model,
        model_state,
        schedule_iterator,
        gt_reward_fn,
        random_trajs,
    ) = initialise(
        model_cls=PreferenceModel,
        rng=ini_rng,
        venv_kwargs=venv_kwargs,
        traj_length=traj_length,
        **initialise_kwargs,
    )

    total_trajs = random_trajs
    recent_agent_trajs = copy(random_trajs)

    train_ds = None

    log.info("Beginning demonstration and preference RLHF")

    for i, agent_train_steps, num_train_prefs in schedule_iterator:
        rng, pref_rng = jrnd.split(rng)

        new_pref_data = get_pref_data(
            pref_fn=partial(oracle_preferences, gt_reward_fn),
            num_prefs=num_train_prefs,
            rng=pref_rng,
            total_trajs=total_trajs,
            recent_trajs=recent_agent_trajs,
            pref_max_old_traj_frac=pref_max_old_traj_frac,
            fragment_length=pref_fragment_length,
        )

        if train_ds is None:
            train_ds = new_pref_data
        else:
            train_ds = {k: jnp.concatenate([v, new_pref_data[k]], axis=0)
                        for k, v in train_ds.items()}

        # Step 3: Train Reward Model

        if (i == 0) and (initial_rm_train_multiplier is not None):
            adjusted_rm_train_kwargs = copy(rm_train_kwargs)
            adjusted_rm_train_kwargs["train_epochs"] *= initial_rm_train_multiplier
        else:
            adjusted_rm_train_kwargs = rm_train_kwargs

        rng, train_rng = jrnd.split(rng)

        log.info("Training reward model")
        model_state = train_reward_model(
            model_state=model_state,
            train_ds=train_ds,
            device_info=device_info,
            rng=train_rng,
            **adjusted_rm_train_kwargs,
        )

        learnt_reward_fn = get_reward_fn(model, model_state.params)
        agent = update_reward_fn(agent, learnt_reward_fn)

        # Step 4: Train agent and record rollouts

        log.info("Training agent")
        traj_extractor = TrajectoryExtractorCallback()
        agent.learn(callback=traj_extractor,
                    total_timesteps=agent_train_steps, **agent_train_kwargs)

        new_trajs = traj_extractor.empty()
        recent_agent_trajs = filter_trim_in_iterable(new_trajs, traj_length)
        total_trajs.extend(recent_agent_trajs)
        log.info(f"{len(new_trajs)=} {len(recent_agent_trajs)=} {len(total_trajs)=}")

        # See how well agent is doing
        if num_to_visualise_per_iteration is not None:
            visualise_policy(
                venv2env_kwargs(venv_kwargs),
                agent,
                attempts=num_to_visualise_per_iteration,
            )

    return agent, (model, model_state), total_trajs


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO",
        format='%(asctime)s %(module)-10s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    for name in ["absl", "jax"]:
        logging.getLogger(name).setLevel("WARNING")

    rlhf(
        traj_length=200,
        pref_fragment_length=10,
        initialise_kwargs={
            "gt_reward_fn": "cartpole",
            "schedule_kwargs": {
                "rlhf_iters": 8,
                "env_timestep_budget": 512_000,
                "pref_comparison_budget": 120,
            },
            "reward_predictor_spec": "Ensemble(16, MLP([32, 32, 1]), True)",
            "agent_kwargs": {"n_steps": 512},
        },

        venv_kwargs={
            "env_id": "CustomCartPole-v0",
            "n_envs": 4,
            "vec_env_cls": "SubprocVecEnv",
        },

        rm_train_kwargs={
            "train_epochs": 2,
            "batch_size": 8,
            "steps_per_eval": 1,
        },
        agent_train_kwargs={"log_interval": 4},
        initial_rm_train_multiplier=4,
        num_to_visualise_per_iteration=2,
        pref_max_old_traj_frac=0.2,
    )

