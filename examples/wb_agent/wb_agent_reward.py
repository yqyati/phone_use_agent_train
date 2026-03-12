"""Custom reward function for wb_agent GRPO training.

The score comes directly from the InfiniteWeb task evaluator (0.0–1.0),
collected step-by-step via WebWorldInteraction.generate_response().
We return the final step's score, matching AndroidWorld's reward convention.
"""


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """Return the web task score from the last rollout step.

    Args:
        data_source: Should be "wb_agent".
        solution_str: Decoded response text (not used; score comes from env).
        ground_truth: website_id string (for bookkeeping only).
        extra_info: Contains "turn_scores" — list of per-step env scores
                    collected by WebWorldInteraction.generate_response().

    Returns:
        Final task score in [0.0, 1.0].  1.0 = all target items found in localStorage.
    """
    if extra_info is None:
        return 0.0

    turn_scores = extra_info.get("turn_scores", [])
    if not turn_scores:
        return 0.0

    return float(turn_scores[-1])
