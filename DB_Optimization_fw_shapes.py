from ema_workbench.em_framework.optimization import EpsilonProgress, ArchiveLogger
from ema_workbench import (
    Model,
    Policy,
    ema_logging,
    SequentialEvaluator,
    MultiprocessingEvaluator,
    MPIEvaluator,
    perform_experiments,
    Samplers,
    Scenario,
    ScalarOutcome
)
import os
from problem_formulation import get_model_for_problem_formulation
import pandas as pd
import os

if __name__ == '__main__':
    ema_logging.log_to_stderr(level=ema_logging.INFO)

    dike_model, planning_steps = get_model_for_problem_formulation(2)
    seed = 27097
    epsilon = [0.1]* len(dike_model.outcomes)
    nfe = 50000  # Set to your actual target
    fw_shapes = [0, 22, 44, 66, 88, 110, 132]

    for fw_shape in fw_shapes:
        print(f"\nRunning flood wave shape {fw_shape}")

        reference_scenario = Scenario('reference',
                                      **{'discount rate 0': 1.5, 'discount rate 1': 1.5, 'discount rate 2': 1.5,
                                         'A.0_ID flood wave shape': fw_shape,
                                         'A.1_Bmax': 190.0, 'A.1_pfail': 0.6720691719092429, 'A.1_Brate': 1.0,
                                         'A.2_Bmax': 245.05131609737873,
                                         'A.2_pfail': 0.5, 'A.2_Brate': 1.0, 'A.3_Bmax': 190.0, 'A.3_pfail': 0.5,
                                         'A.3_Brate': 1.0,
                                         'A.4_Bmax': 190.0, 'A.4_pfail': 0.5, 'A.4_Brate': 1.0, 'A.5_Bmax': 190.0,
                                         'A.5_pfail': 0.5,
                                         'A.5_Brate': 1.0})

        with MultiprocessingEvaluator(dike_model) as evaluator:

            logger = ArchiveLogger(
            r"output_oscar/fw_shapes",
            decision_varnames=[l.name for l in dike_model.levers],
            outcome_varnames=[o.name for o in dike_model.outcomes],
            base_filename=f"seed_{seed}_fw_shape_{fw_shape}_archive.tar.gz",
            )

            convergence_metrics = [logger, EpsilonProgress()]


            results, convergence = evaluator.optimize(
                nfe=nfe,
                searchover="levers",
                epsilons=epsilon,
                convergence=convergence_metrics,
                reference= reference_scenario,
                seed=seed,
            )

            # Save regular results
            pd.DataFrame(results).to_csv(f"output_oscar/fw_shapes/results_seed_{seed}_fw_shape_{fw_shape}.csv", index=False)
            pd.DataFrame(convergence).to_csv(f"output_oscar/fw_shapes/convergence_seed_{seed}_fw_shape_{fw_shape}.csv", index=False)

            # Extract archive logger contents
            #logger_results = logger.load()
            #pd.DataFrame(logger_results).to_csv(f"output/archive_seed_{seed}.csv", index=False)

            print(f"Saved all outputs for flood wave shape {fw_shape}")
