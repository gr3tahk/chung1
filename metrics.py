def lcs_length(seq_a, seq_b):
    rows = len(seq_a) + 1
    cols = len(seq_b) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i, token_a in enumerate(seq_a, start=1):
        for j, token_b in enumerate(seq_b, start=1):
            if token_a == token_b:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def compute_tes(executed, reference, beta=1.0):
    if not executed and not reference:
        return 1.0
    if not executed or not reference:
        return 0.0
    lcs = lcs_length(executed, reference)
    precision = lcs / len(executed)
    recall = lcs / len(reference)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta ** 2
    return ((1 + beta_sq) * precision * recall) / ((beta_sq * precision) + recall)


def compute_ites(executed, reference, beta=1.0):
    if not executed:
        return 0.0
    prefix_scores = []
    for index in range(1, len(executed) + 1):
        prefix_scores.append(compute_tes(executed[:index], reference, beta=beta))
    return sum(prefix_scores) / len(prefix_scores)


def score_against_references(executed, reference_trajectories):
    best = {
        "best_reference_id": None,
        "tes": 0.0,
        "ites": 0.0,
    }
    for reference in reference_trajectories:
        tes = compute_tes(executed, reference["actions"])
        ites = compute_ites(executed, reference["actions"])
        candidate = {
            "best_reference_id": reference["id"],
            "tes": tes,
            "ites": ites,
        }
        if (tes, ites) > (best["tes"], best["ites"]):
            best = candidate
    return best
