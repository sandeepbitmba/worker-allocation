# Contextual Combinatorial Upper Confidence Bound (CC-CUCB) for Adaptive Worker Selection

## Overview

This repository contains the Python implementation and supplementary experimental results for the research study **“A Contextual Combinatorial Upper Confidence Bound Framework for Adaptive Worker Selection in Crowdsourcing.”**

The study develops a contextual bandit-based framework for adaptive worker selection in crowdsourcing environments. The proposed Contextual Combinatorial Upper Confidence Bound (CC-CUCB) algorithm estimates worker-specific reward parameters using regularised linear regression and selects the top-\(K\) eligible workers based on contextual upper confidence bound scores.

The framework is evaluated using the NYT Crowdsourcing Topical Relevance dataset, with multiple baseline algorithms and repeated experimental runs.

## Research Objectives

The implementation is designed to:

- Develop a contextual approach to adaptive worker selection.
- Estimate worker-specific reward parameters using ridge regression.
- Incorporate contextual uncertainty into worker-selection decisions.
- Compare CC-CUCB against alternative worker-selection strategies.
- Evaluate cumulative reward and cumulative regret.
- Examine the effect of different worker-selection group sizes.
- Assess performance using independent experimental runs and statistical tests.

## Proposed Method: CC-CUCB

For each task \(t\), the algorithm observes a contextual feature vector \(x_t\) and identifies the eligible worker set.

For each worker \(j\), the regularised design matrix and reward-weighted context vector are updated as follows:

\[
A_j(t)=\lambda I+\sum_{\tau<t}x_\tau x_\tau^T
\]

\[
b_j(t)=\sum_{\tau<t}r_{\tau,j}x_\tau
\]

The worker-specific parameter estimate is:

\[
\hat{\theta}_j(t)=A_j(t)^{-1}b_j(t)
\]

The contextual upper confidence bound score is:

\[
UCB_{t,j}=x_t^T\hat{\theta}_j(t)+
\alpha\sqrt{x_t^TA_j(t)^{-1}x_t}
\]

The algorithm selects the \(K\) eligible workers with the highest UCB scores.

The inverse design matrices are updated efficiently using the Sherman–Morrison rank-one update.

## Dataset

The experiment uses the **Pilot 3P-Doc-High** subset from the NYT Crowdsourcing Topical Relevance dataset.

**Dataset source:**

Inel, O., Khamkham, K., Craswell, N., & others (2018). NYT Crowdsourcing Topical Relevance dataset.

Zenodo DOI: https://doi.org/10.5281/zenodo.1478495

### Experimental dataset characteristics

| Parameter | Value |
|---|---:|
| Distinct workers | 44 |
| Distinct tasks | 60 |
| Observed annotations | 900 |
| Eligible workers per task | 15 |
| Selected workers (\(K\)) | 5 |
| Context dimensions (\(d\)) | 12 |
| Experimental runs | 30 |
| Decision horizon | 500 |

### Reward coding

The original relevance judgements are converted into binary rewards:

- `highly_relevant` = 1
- `relevant` = 1
- `not_relevant` = 0

Only observed worker–task pairs are treated as eligible. Missing worker–task combinations are not imputed as zero-reward observations.

### Context construction

The task-context vector includes:

- Intercept term.
- Topic-level one-hot encoded features.
- Standardised logarithm of document length.

## Algorithms Compared

The implementation evaluates the following strategies:

1. **Random:** Random selection of eligible workers.
2. **Epsilon-greedy:** Exploration and exploitation using an epsilon-greedy selection strategy.
3. **CUCB:** Non-contextual confidence-bound-based worker selection.
4. **LinUCB-single:** Contextual upper confidence bound selection of a single worker.
5. **Diagonal Contextual Thompson Sampling:** Contextual Thompson sampling using a diagonal Gaussian approximation.
6. **CC-CUCB:** Proposed contextual combinatorial upper confidence bound algorithm.
7. **Oracle:** Task-specific hindsight oracle selecting the highest-reward eligible workers.

The LinUCB-single baseline selects one worker per decision, whereas the combinatorial strategies select \(K\) workers.

## Experimental Configuration

| Parameter | Setting |
|---|---:|
| Number of independent runs | 30 |
| Decision horizon | 500 |
| Default selection size (\(K\)) | 5 |
| Exploration coefficient (\(\alpha\)) | 1.0 |
| Ridge regularisation (\(\lambda\)) | 1.0 |
| Epsilon-greedy exploration rate | 0.10 |
| Random seed base | 20261121 |

The same task sequence is used across competing methods within each run to support paired comparisons. Independent random seeds are used for the policy-specific stochastic decisions.

## Performance Evaluation

The framework evaluates performance using:

- Cumulative reward.
- Mean reward per decision.
- Cumulative regret relative to the task-specific hindsight oracle.
- Mean and standard deviation across independent runs.
- 95% confidence intervals for cumulative reward and regret.
- Friedman test for overall comparisons across online methods.
- Wilcoxon signed-rank tests for paired comparisons.
- Cohen's \(d_z\) for paired effect-size estimation.
- Group-size sensitivity analysis.

The task-specific hindsight oracle provides an upper-bound reference under the observed reward information. It is not an implementable online policy.

## Repository Structure

```text
worker-allocation/
│
├── cc_cucb_experiment.py
│
├── Supplementary_Results/
│   ├── Supplementary experimental outputs
│   └── Supporting result files
│
└── README.md
```

The experiment script expects the prepared input dataset to be available separately as `pilot_3p_doc_high_900_annotations.csv`.

## Requirements

The implementation requires Python 3.9 or later and the following libraries:

```bash
pip install numpy pandas scipy
```

## Reproducibility Instructions

### Step 1: Clone the repository

```bash
git clone https://github.com/sandeepbitmba/worker-allocation.git
cd worker-allocation
```

### Step 2: Prepare the dataset

Obtain the NYT Crowdsourcing Topical Relevance archive from:

https://doi.org/10.5281/zenodo.1478495

Prepare the Pilot 3P-Doc-High subset as:

```text
pilot_3p_doc_high_900_annotations.csv
```

The prepared CSV must contain the following columns:

| Column | Description |
|---|---|
| `task_id` | Unique task identifier |
| `worker_id` | Unique worker identifier |
| `topic_id` | Task topic identifier |
| `document_length` | Document length |
| `binary_reward` | Binary relevance reward (0 or 1) |

Place the prepared CSV in the repository's working directory.

### Step 3: Execute the experiment

```bash
python cc_cucb_experiment.py
```

Alternatively, specify the input dataset and output directory:

```bash
python cc_cucb_experiment.py --data pilot_3p_doc_high_900_annotations.csv --out cc_cucb_outputs
```

### Step 4: Inspect the generated results

The script automatically generates the following experimental output files:

| Output file | Description |
|---|---|
| `per_run_results.csv` | Performance results for each method across 30 runs |
| `performance_summary.csv` | Mean, standard deviation, and confidence intervals |
| `paired_statistical_tests.csv` | Wilcoxon signed-rank tests and paired effect sizes |
| `friedman_test.csv` | Overall Friedman test results |
| `cumulative_curves_long.csv` | Run-level cumulative reward and regret trajectories |
| `group_size_sensitivity.csv` | Sensitivity analysis for different values of \(K\) |

## Computational Considerations

The implementation maintains worker-specific inverse design matrices and updates them through the Sherman–Morrison formula. This avoids repeatedly computing full matrix inverses after every observation.

The computational cost depends primarily on the number of workers, contextual feature dimensions, decision horizon, and selected worker group size.

## Citation

If you use this implementation in academic research, please cite the associated research paper:

**Bhattacharjee, S.** (Manuscript under review). *A Contextual Combinatorial Upper Confidence Bound Framework for Adaptive Worker Selection in Crowdsourcing.*

Repository: https://github.com/sandeepbitmba/worker-allocation

## Reproducibility and Research Transparency

The repository provides the Python implementation and supplementary experimental materials supporting the computational evaluation of the proposed CC-CUCB framework. The experiment uses a publicly available crowdsourcing dataset and fixed random-seed settings to facilitate reproducibility.

The prepared input CSV is required separately and is not generated automatically by the experiment script. Researchers should retain the original dataset attribution and document any preprocessing changes when reproducing or extending the experiments.

## License

Please contact the repository owner for permission and licensing information before redistributing or commercially reusing the research code.
