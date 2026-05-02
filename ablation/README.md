# Additional Prediction Experiments

This folder contains the additional prompt-based prediction experiments run for
GPT, DeepSeek, and Gemini. Each experiment has a separate entry-point script.

Experiments:

- `cot` - hidden chain-of-thought style prompting.
- `story_facts` - facts/story-only prompting.

Examples:

```bash
python ablation/run_gpt_cot.py \
  --data_path data/final_dataset.csv \
  --output_dir results/ablation/gpt_cot

python ablation/run_deepseek_story_facts.py \
  --data_path data/final_dataset.csv \
  --output_dir results/ablation/deepseek_story_facts

python ablation/run_gemini_story_facts.py \
  --data_path data/final_dataset.csv \
  --output_dir results/ablation/gemini_story_facts
```

Generated outputs should be written under `results/` and are ignored by git.
