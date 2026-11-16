"""M6: offline ranking evaluation and (from M6c) the learned ranker.

- ``dataset``: logged ranking pools joined with hand labels, one feature dict per candidate.
- ``metrics``: NDCG@10 / P@10 / bad@10 on labeled pools, v0 and simple baselines, bootstrap
  intervals over current videos, time-ordered and leave-one-seed-out splits.
- ``just eval`` (``__main__``): prints all of it; ``--csv`` exports the dataset for notebooks.
"""
