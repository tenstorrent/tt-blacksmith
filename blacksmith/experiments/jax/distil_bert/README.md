# DistilBERT Knowledge Distillation Experiment

This directory contains the code for performing **knowledge distillation** of a DistilBERT model on the **SST-2** sentiment classification dataset using **TT-XLA**.
The goal of this experiment is to train a smaller *student* model (DistilBERT) to mimic a larger *teacher* model (BERT-base) while maintaining high performance on sentiment classification.
Original paper can be found [here](https://arxiv.org/abs/1910.01108)

---

## Overview

The experiment applies **knowledge distillation** to compress the `textattack/bert-base-uncased-SST-2` model into a smaller `distilbert-base-uncased` model.
Distillation combines several loss components — *cross-entropy*, *Kullback-Leibler divergence (KL)*, and *cosine similarity* — to balance the supervision from the ground-truth labels and the teacher’s soft predictions.

Training is implemented in **Flax (JAX)** and designed to run on **TT devices** via **TT-XLA**.

---

## Training

```bash
python3 blacksmith/experiments/jax/distil_bert/train_distillation.py
```

---

## Data

The **GLUE SST-2 (Stanford Sentiment Treebank 2)** dataset is a benchmark for **binary sentiment classification**.
Each sample consists of a movie review labeled as *positive* or *negative* sentiment.

Source: [Hugging Face Dataset Hub](https://huggingface.co/datasets/glue/viewer/sst2)

### Example
```json
{
  "sentence": "A touching and insightful film.",
  "label": 1
}
```
- **sentence**: A short movie review or phrase.
- **label**: Sentiment label (1 = positive, 0 = negative).

---

## Configuration

The experiment parameters are defined in `test_distil_bert_flax.yaml`.
This configuration specifies dataset, model settings, training hyperparameters, and loss weighting used during distillation.

The current configuration reflects tested and recommended defaults for the SST-2 experiment.

---

### Configuration Parameters

| **Category** | **Parameter** | **Description** | **Default Value** |
|---------------|---------------|-----------------|-------------------|
| **Dataset** | `dataset_id` | Dataset identifier on Hugging Face. | `"glue/sst2"` |
|  | `tokenizer_name` | Tokenizer used for encoding text. | `"bert-base-uncased"` |
|  | `max_length` | Maximum tokenized input length. | `128` |
| **Model** | `teacher_model` | Pre-trained teacher model used for distillation. | `"textattack/bert-base-uncased-SST-2"` |
|  | `student_model` | Student model to be trained. | `"distilbert-base-uncased"` |
|  | `dtype` | Data type for computation. | `"jax.bfloat16"` |
| **Training** | `learning_rate` | Optimizer learning rate. | `1e-5` |
|  | `batch_size` | Number of samples per batch. | `32` |
|  | `num_epochs` | Total number of training epochs. | `3` |
|  | `weight_decay` | Weight decay coefficient. | `0.01` |
|  | `warmup_ratio` | Ratio of warmup steps to total training steps. | `0.06` |
|  | `optimizer` | Optimizer used for training. | `"adamw"` |
|  | `seed` | Random seed for reproducibility. | `42` |
|  | `resume_from_checkpoint` | If `False`, training starts fresh and deletes old checkpoints. | `False` |
| **Loss** | `temperature` | Softening temperature for distillation logits. | `2.0` |
|  | `alpha_kl` | Weight for KL-divergence loss. | `0.45` |
|  | `alpha_ce` | Weight for cross-entropy loss. | `1.0` |
|  | `alpha_cos` | Weight for cosine similarity loss. | `0.1` |
| **Logging** | `use_wandb` | Enable or disable Weights & Biases logging. | `False` |
|  | `experiment_name` | Name of this specific experiment. | `"Flax DistilBERT on SST-2"` |
|  | `project_name` | W&B project name. | `"bert-distillation"` |
|  | `job_name` | Identifier for this run. | `"distillation"` |
|  | `log_every` | Steps between logging training metrics. | `100` |
|  | `log_val_every` | Steps between validation runs. | `100` |
|  | `do_checkpoint` | Whether to save model checkpoints. | `False` |
|  | `checkpoint_every` | Save frequency in steps. | `100` |
|  | `keep_top_k_checkpoints` | Number of best checkpoints to keep. | `2` |
|  | `output_dir` | Directory for experiment outputs. | `"blacksmith/experiments/jax/distil_bert"` |

---
