"""Custom reward function for AndroidWorld GRPO training.

This function is called by verl's NaiveRewardManager after each episode.
The reward comes directly from the AndroidWorld task evaluator (0.0~1.0),
collected step-by-step in `turn_scores` during the rollout.

Since AndroidWorld provides environment-side evaluation (not text-based),
we simply take the final step's score rather than re-evaluating from text.
"""


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """Return the AndroidWorld task score from the last rollout step.

    Args:
        data_source: Should be "android_world".
        solution_str: Decoded response text (not used; score comes from env).
        ground_truth: Task type string (for bookkeeping only).
        extra_info: Contains "turn_scores" — list of per-step env scores
                    collected by AndroidWorldInteraction.generate_response().

    Returns:
        Final task score in [0.0, 1.0].  1.0 = fully completed.
    """
    if extra_info is None:
        return 0.0

    turn_scores = extra_info.get("turn_scores", [])
    if not turn_scores:
        return 0.0

    # Use the last score in the episode (after the final action or max_steps)
    return float(turn_scores[-1])
