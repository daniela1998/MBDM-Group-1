from ema_workbench import (
    ema_logging,
    MultiprocessingEvaluator,
)
from ema_workbench.em_framework.optimization import EpsilonProgress, ArchiveLogger
from problem_formulation import get_model_for_problem_formulation
import pandas as pd
import os

if __name__ == '__main__':
    ema_logging.log_to_stderr(level=ema_logging.INFO)

    dike_model, planning_steps = get_model_for_problem_formulation(2)
    seeds = [27097, 367, 9886, 61, 578986]
    epsilon = [0.1] * len(dike_model.outcomes)
    nfe = 1000  # Set to your actual target
    os.makedirs("output", exist_ok=True)

    for seed in seeds:
        print(f"\nRunning seed {seed}")

        # Use a unique name for each archive log file
        archive_filename = f"output/seed_{seed}_archive.tar.gz"
        logger = ArchiveLogger(
            archive_filename,
            decision_varnames=[l.name for l in dike_model.levers],
            outcome_varnames=[o.name for o in dike_model.outcomes],
        )
        convergence_metrics = [logger, EpsilonProgress()]

        with MultiprocessingEvaluator(dike_model) as evaluator:
            results, convergence = evaluator.optimize(
                nfe=nfe,
                searchover="levers",
                epsilons=epsilon,
                convergence=convergence_metrics,
                reference={},
                seed=seed,
            )

        # Save regular results
        pd.DataFrame(results).to_csv(f"output/results_seed_{seed}.csv", index=False)
        pd.DataFrame(convergence).to_csv(f"output/convergence_seed_{seed}.csv", index=False)

        # Extract archive logger contents
        logger_results = logger.load()
        pd.DataFrame(logger_results).to_csv(f"output/archive_seed_{seed}.csv", index=False)

        print(f"Saved all outputs for seed {seed}")

