# Source: B_SpiralDance

This policy folder is derived from the **G1 Moves** dataset.

| | |
|---|---|
| Dataset | [exptech/g1-moves](https://huggingface.co/datasets/exptech/g1-moves), revision `895d064b2385725e6aabf540461c219afc3e0916` |
| Clip | `dance/B_SpiralDance` |
| Performer | Mitch Chaiet |
| License | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) |

## Files and changes

- `params/B_SpiralDance.csv`: the dataset's `retarget/B_SpiralDance.csv`, unchanged.
- `exported/policy.onnx`: the dataset's `policy_154/B_SpiralDance_policy.onnx`, reduced to the `obs` → `actions` subgraph. The original also takes a `time_step` input, which g1_ctrl doesn't feed.
- `params/deploy.yaml`: generated from the ONNX metadata (joint order, PD gains, default pose, action scale).

Converted for unitree_rl_lab `g1_ctrl` (State_Mimic) with `g1moves_to_g1ctrl.py` from the team's g1-workspace repo.

## Citation

```bibtex
@misc{g1moves2026,
  title     = {G1 Moves: Motion Capture Dataset for the Unitree G1 Humanoid Robot},
  author    = {Chaiet, Mitch},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/exptech/g1-moves},
  note      = {60 motion capture clips with retargeted joint trajectories, RL training data, and trained policies}
}
```
