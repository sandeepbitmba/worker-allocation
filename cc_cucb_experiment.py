"""
Reproducible CC-CUCB replay experiment.
Input: pilot_3p_doc_high_900_annotations.csv
Derived from Pilot 3P-Doc-High/f1200503_anonymized.csv in the NYT
Crowdsourcing Topical Relevance archive (Inel et al., 2018; Zenodo
DOI 10.5281/zenodo.1478495).

Reward coding: relevant and highly_relevant -> 1; not_relevant -> 0.
Only observed worker-task pairs are eligible; missing pairs are never
imputed as zero. Contextual Thompson sampling uses a diagonal Gaussian
approximation to the worker-specific linear posterior.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, t

def build_problem(data):
    task_meta = data.groupby("task_id").agg(
        topic=("topic_id", "first"),
        doc_len=("document_length", "first")
    )
    topic_levels = sorted(task_meta["topic"].unique())
    loglen = np.log1p(task_meta["doc_len"].astype(float))
    mu, sd = loglen.mean(), loglen.std(ddof=0)
    X = np.column_stack([
        np.ones(len(task_meta)),
        *[(task_meta["topic"].values == lev).astype(float) for lev in topic_levels],
        (loglen.values - mu) / (sd if sd else 1.0)
    ])
    task_ids = list(task_meta.index)
    task_index = {tid: i for i, tid in enumerate(task_ids)}
    worker_ids = sorted(data["worker_id"].unique())
    worker_index = {wid: i for i, wid in enumerate(worker_ids)}
    candidates = [[] for _ in task_ids]
    rewards = [dict() for _ in task_ids]
    for row in data.itertuples(index=False):
        ti = task_index[row.task_id]
        wi = worker_index[row.worker_id]
        candidates[ti].append(wi)
        rewards[ti][wi] = float(row.binary_reward)
    assert len(task_ids) == 60 and len(worker_ids) == 44
    assert sum(map(len, candidates)) == 900
    assert all(len(pool) == 15 for pool in candidates)
    return X, task_ids, worker_ids, candidates, rewards

def run_policy(method, seed, task_sequence, X, candidates, rewards,
               n_workers, K=5, alpha=1.0, lam=1.0, epsilon=0.10):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    counts = np.zeros(n_workers)
    reward_sums = np.zeros(n_workers)
    # Maintain A^{-1} directly and apply Sherman-Morrison rank-one updates.
    Ainv = np.asarray([np.eye(d) / lam for _ in range(n_workers)])
    b = np.zeros((n_workers, d))
    cumulative_reward = 0.0
    cumulative_regret = 0.0
    reward_curve, regret_curve = [], []

    for t_idx, task_idx in enumerate(task_sequence):
        x = X[task_idx]
        pool = np.asarray(candidates[task_idx], dtype=int)
        pool_rewards = np.asarray(
            [rewards[task_idx][int(w)] for w in pool], dtype=float
        )
        oracle_topk = float(np.sort(pool_rewards)[-K:].mean())

        if method == "Random":
            selected = rng.choice(pool, size=K, replace=False)
        elif method == "Oracle":
            selected = pool[np.argsort(pool_rewards)[-K:]]
        elif method == "CUCB":
            means = reward_sums[pool] / np.maximum(counts[pool], 1)
            means = np.where(counts[pool] > 0, means, 0.5)
            bonus = np.sqrt(2 * np.log(t_idx + 2) / (counts[pool] + 1))
            selected = pool[np.argsort(means + bonus)[-K:]]
        elif method == "Epsilon-greedy":
            if rng.random() < epsilon:
                selected = rng.choice(pool, size=K, replace=False)
            else:
                means = reward_sums[pool] / np.maximum(counts[pool], 1)
                means = np.where(counts[pool] > 0, means, 0.5)
                selected = pool[np.argsort(means)[-K:]]
        elif method in ("CC-CUCB", "LinUCB-single", "Diagonal Contextual-TS"):
            scores = []
            for worker in pool:
                theta = Ainv[worker] @ b[worker]
                if method == "Diagonal Contextual-TS":
                    # Approximate N(theta, A^{-1}) with independent marginals.
                    theta_sample = theta + rng.normal(size=d) * np.sqrt(
                        np.maximum(np.diag(Ainv[worker]), 0.0)
                    )
                    scores.append(x @ theta_sample)
                else:
                    scores.append(
                        x @ theta + alpha * np.sqrt(
                            max(0.0, x @ Ainv[worker] @ x)
                        )
                    )
            scores = np.asarray(scores)
            if method == "LinUCB-single":
                selected = np.asarray([pool[np.argmax(scores)]])
            else:
                selected = pool[np.argsort(scores)[-K:]]
        else:
            raise ValueError(method)

        selected_rewards = np.asarray(
            [rewards[task_idx][int(w)] for w in selected], dtype=float
        )
        round_reward = float(selected_rewards.mean())

        for worker, reward in zip(selected, selected_rewards):
            counts[worker] += 1
            reward_sums[worker] += reward
            if method in ("CC-CUCB", "LinUCB-single", "Diagonal Contextual-TS"):
                z = Ainv[worker] @ x
                Ainv[worker] -= np.outer(z, z) / (1.0 + x @ z)
                b[worker] += reward * x

        cumulative_reward += round_reward
        if method != "LinUCB-single":
            cumulative_regret += oracle_topk - round_reward
        reward_curve.append(cumulative_reward)
        regret_curve.append(cumulative_regret)

    return {
        "final_reward": cumulative_reward,
        "mean_reward": cumulative_reward / len(task_sequence),
        "cum_regret": np.nan if method == "LinUCB-single" else cumulative_regret,
        "rewards": np.asarray(reward_curve),
        "regrets": np.asarray(regret_curve)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="pilot_3p_doc_high_900_annotations.csv")
    parser.add_argument("--out", default="cc_cucb_outputs")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data)
    X, task_ids, worker_ids, candidates, rewards = build_problem(data)
    methods = ["Random", "Epsilon-greedy", "CUCB", "LinUCB-single",
               "Diagonal Contextual-TS", "CC-CUCB", "Oracle"]
    n_runs, horizon, K = 30, 500, 5
    records, curves = [], {}

    for run in range(n_runs):
        base_seed = 20261121 + run
        task_sequence = np.random.default_rng(base_seed).integers(
            0, len(task_ids), size=horizon
        )
        for method_index, method in enumerate(methods):
            result = run_policy(
                method, base_seed + 1000 * (method_index + 1),
                task_sequence, X, candidates, rewards, len(worker_ids),
                K=K, alpha=1.0, lam=1.0, epsilon=0.10
            )
            records.append({
                "run": run + 1, "seed": base_seed, "method": method,
                "final_reward": result["final_reward"],
                "mean_reward": result["mean_reward"],
                "cum_regret": result["cum_regret"]
            })
            curves[(run, method)] = result

    results = pd.DataFrame(records)
    results.to_csv(out / "per_run_results.csv", index=False)
    online = ["Random", "Epsilon-greedy", "CUCB",
              "LinUCB-single", "Diagonal Contextual-TS", "CC-CUCB"]
    wide = results.pivot(index="run", columns="method", values="final_reward")
    friedman = friedmanchisquare(*[wide[m].values for m in online])

    summary = results.groupby("method").agg(
        mean_cumulative_reward=("final_reward", "mean"),
        sd_cumulative_reward=("final_reward", "std"),
        mean_reward_per_round=("mean_reward", "mean"),
        mean_cumulative_regret=("cum_regret", "mean"),
        sd_cumulative_regret=("cum_regret", "std")
    ).reindex(methods)
    summary["reward_ci_low"] = summary["mean_cumulative_reward"] - (
        t.ppf(0.975, n_runs-1) * summary["sd_cumulative_reward"] / np.sqrt(n_runs)
    )
    summary["reward_ci_high"] = summary["mean_cumulative_reward"] + (
        t.ppf(0.975, n_runs-1) * summary["sd_cumulative_reward"] / np.sqrt(n_runs)
    )
    summary["regret_ci_low"] = summary["mean_cumulative_regret"] - (
        t.ppf(0.975, n_runs-1) * summary["sd_cumulative_regret"] / np.sqrt(n_runs)
    )
    summary["regret_ci_high"] = summary["mean_cumulative_regret"] + (
        t.ppf(0.975, n_runs-1) * summary["sd_cumulative_regret"] / np.sqrt(n_runs)
    )
    summary.to_csv(out / "performance_summary.csv")

    tests = []
    for comparator in ["Random", "Epsilon-greedy", "CUCB",
                       "LinUCB-single", "Diagonal Contextual-TS", "Oracle"]:
        x, y = wide["CC-CUCB"].values, wide[comparator].values
        W, p = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        diff = x - y
        tests.append([comparator, W, p, diff.mean(),
                      diff.std(ddof=1), diff.mean()/diff.std(ddof=1)])
    pd.DataFrame(tests, columns=[
        "Comparator", "W", "p_value", "Mean_paired_difference",
        "SD_paired_difference", "Cohen_dz"
    ]).to_csv(out / "paired_statistical_tests.csv", index=False)
    pd.DataFrame([{
        "Friedman_chi_square": friedman.statistic,
        "df": len(online)-1, "p_value": friedman.pvalue
    }]).to_csv(out / "friedman_test.csv", index=False)

    curve_rows = []
    for run in range(n_runs):
        for method in methods:
            for round_no, (reward, regret) in enumerate(zip(
                curves[(run, method)]["rewards"],
                curves[(run, method)]["regrets"]
            ), start=1):
                curve_rows.append([run+1, method, round_no, reward, regret])
    pd.DataFrame(curve_rows, columns=[
        "run", "method", "round", "cumulative_reward", "cumulative_regret"
    ]).to_csv(out / "cumulative_curves_long.csv", index=False)

    sensitivity = []
    for K_value in [3, 5, 10, 15]:
        values = []
        for run in range(n_runs):
            seed = 20261121 + run
            sequence = np.random.default_rng(seed).integers(
                0, len(task_ids), size=horizon
            )
            result = run_policy(
                "CC-CUCB", seed + 6000, sequence, X, candidates, rewards,
                len(worker_ids), K=K_value, alpha=1.0, lam=1.0
            )
            values.append([result["final_reward"], result["mean_reward"],
                           result["cum_regret"]])
        arr = np.asarray(values)
        sensitivity.append([
            K_value, arr[:,0].mean(), arr[:,0].std(ddof=1),
            arr[:,1].mean(), arr[:,2].mean(), arr[:,2].std(ddof=1)
        ])
    pd.DataFrame(sensitivity, columns=[
        "K", "Mean_cumulative_reward", "SD_cumulative_reward",
        "Mean_reward_per_round", "Mean_cumulative_regret", "SD_cumulative_regret"
    ]).to_csv(out / "group_size_sensitivity.csv", index=False)

if __name__ == "__main__":
    main()
