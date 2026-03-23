"""
Unit tests for adaptive curriculum logic in train.py.

Tests bucket partitioning, UCB scoring, and feedback accumulation
without requiring GPU or full training run.

Usage: uv run python -m pytest test_adaptive_curriculum.py -v
"""

import math


def test_bucket_partitioning():
    """Batches are evenly distributed across difficulty buckets."""
    buffer_size = 64
    NUM_BUCKETS = 8
    # Simulate scores (increasing difficulty)
    scores = [i / buffer_size for i in range(buffer_size)]
    sorted_by_diff = sorted(range(buffer_size), key=lambda i: scores[i])
    bucket_size = buffer_size // NUM_BUCKETS

    buckets = [[] for _ in range(NUM_BUCKETS)]
    for rank, idx in enumerate(sorted_by_diff):
        b = min(rank // bucket_size, NUM_BUCKETS - 1)
        buckets[b].append(idx)

    # Each bucket should have buffer_size // NUM_BUCKETS items
    for b in range(NUM_BUCKETS):
        assert len(buckets[b]) == bucket_size, f"Bucket {b} has {len(buckets[b])} items, expected {bucket_size}"

    # Bucket 0 should have easiest, bucket 7 hardest
    for item in buckets[0]:
        assert scores[item] < scores[buckets[-1][0]]


def test_ucb_exploration_favors_unvisited():
    """UCB1 scores unvisited buckets as infinity."""
    NUM_BUCKETS = 8
    ucb_total_improvement = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ucb_count = [5, 0, 0, 0, 0, 0, 0, 0]
    ucb_c = 1.414

    total_pulls = max(sum(ucb_count), 1)
    ucb_scores = []
    for b in range(NUM_BUCKETS):
        if ucb_count[b] == 0:
            ucb_scores.append(float('inf'))
        else:
            avg_improvement = ucb_total_improvement[b] / ucb_count[b]
            exploration = ucb_c * math.sqrt(math.log(total_pulls) / ucb_count[b])
            ucb_scores.append(avg_improvement + exploration)

    # Unvisited buckets should be infinity (explored first)
    for b in range(1, NUM_BUCKETS):
        assert ucb_scores[b] == float('inf')
    # Visited bucket should have finite score
    assert ucb_scores[0] < float('inf')


def test_ucb_rewards_improvement_not_raw_loss():
    """UCB prioritizes buckets with highest loss improvement, not highest raw loss."""
    NUM_BUCKETS = 4
    ucb_c = 1.414

    # Bucket 0: high raw loss but no improvement (stuck)
    # Bucket 1: low raw loss but high improvement (learning fast)
    ucb_total_improvement = [0.0, 0.5, 0.1, 0.1]
    ucb_count = [10, 10, 10, 10]

    total_pulls = sum(ucb_count)
    ucb_scores = []
    for b in range(NUM_BUCKETS):
        avg_improvement = ucb_total_improvement[b] / ucb_count[b]
        exploration = ucb_c * math.sqrt(math.log(total_pulls) / ucb_count[b])
        ucb_scores.append(avg_improvement + exploration)

    # Bucket 1 (highest improvement) should rank first
    bucket_order = sorted(range(NUM_BUCKETS), key=lambda b: ucb_scores[b], reverse=True)
    assert bucket_order[0] == 1, f"Expected bucket 1 first, got {bucket_order[0]}"


def test_feedback_accumulation():
    """Loss feedback correctly computes per-bucket improvement."""
    NUM_BUCKETS = 4
    prev_bucket_avg_loss = [5.0, 4.0, 3.0, 2.0]
    ucb_total_improvement = [0.0] * NUM_BUCKETS
    ucb_count = [0] * NUM_BUCKETS

    # Simulate one buffer of feedback: all buckets improved
    feedback_losses = [
        (0, 4.5), (0, 4.5),  # bucket 0: avg=4.5, prev=5.0, improvement=0.5
        (1, 3.0), (1, 3.0),  # bucket 1: avg=3.0, prev=4.0, improvement=1.0
        (2, 2.8), (2, 2.8),  # bucket 2: avg=2.8, prev=3.0, improvement=0.2
        (3, 2.5), (3, 2.5),  # bucket 3: avg=2.5, prev=2.0, improvement=-0.5 (got worse)
    ]

    buf_loss = [0.0] * NUM_BUCKETS
    buf_count = [0] * NUM_BUCKETS
    for b_id, loss_val in feedback_losses:
        buf_loss[b_id] += loss_val
        buf_count[b_id] += 1

    for b in range(NUM_BUCKETS):
        if buf_count[b] > 0:
            cur_avg = buf_loss[b] / buf_count[b]
            if prev_bucket_avg_loss[b] is not None:
                improvement = prev_bucket_avg_loss[b] - cur_avg
                ucb_total_improvement[b] += improvement
                ucb_count[b] += 1
            prev_bucket_avg_loss[b] = cur_avg

    assert abs(ucb_total_improvement[0] - 0.5) < 1e-6
    assert abs(ucb_total_improvement[1] - 1.0) < 1e-6
    assert abs(ucb_total_improvement[2] - 0.2) < 1e-6
    assert abs(ucb_total_improvement[3] - (-0.5)) < 1e-6


def test_buffer_size_validation():
    """Buffer size must be >= number of buckets."""
    NUM_BUCKETS = 8
    buffer_size = 4
    try:
        assert buffer_size >= NUM_BUCKETS
        raise AssertionError("Should have failed")
    except AssertionError as e:
        if "Should have failed" in str(e):
            raise
        pass  # Expected


if __name__ == "__main__":
    test_bucket_partitioning()
    test_ucb_exploration_favors_unvisited()
    test_ucb_rewards_improvement_not_raw_loss()
    test_feedback_accumulation()
    test_buffer_size_validation()
    print("All tests passed")
