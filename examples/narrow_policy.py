"""Trusted local policy plugin.

Load it with seb_dream_rsi.policy.load_policy_file. The runner imports this
module in-process; write the file yourself rather than pointing the loader
at model output.
"""

from seb_dream_rsi.policy import ParallelRefinePolicy, PolicySpec

POLICY = ParallelRefinePolicy(
    PolicySpec(
        family="parallel_refine",
        branch_count=2,
        refine_depth=2,
        name="narrow-parallel-refine",
    )
)
