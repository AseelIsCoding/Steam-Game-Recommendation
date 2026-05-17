[README.md](https://github.com/user-attachments/files/27909880/README.md)
# Steam Game Recommendation with LLM-Generated Tag Descriptions

<p align="center">
  <img src="assets/pacman.gif" width="650" alt="Pac-Man animation">
</p>

<p align="center">
  <b>A neural recommender system for Steam games using NRMS and LLM-generated tag descriptions.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=flat-square">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=flat-square">
  <img src="https://img.shields.io/badge/Transformers-DistilBERT-yellow?style=flat-square">
  <img src="https://img.shields.io/badge/Task-Recommender%20Systems-green?style=flat-square">
  <img src="https://img.shields.io/badge/LLM-Tag%20Descriptions-purple?style=flat-square">
</p>

---

## Overview

This project explores whether **LLM-generated tag descriptions** can improve Steam game recommendation.

Steam has thousands of games, and users usually interact with only a small part of the catalogue. A good recommender system can help users discover games that better match their interests.

In this project, we adapted an idea from news recommendation to the gaming domain. Instead of using news titles and category descriptions, we represent Steam games using their titles, tags, and short LLM-generated descriptions of those tags.

---

## Project Goal

The main goal is to test whether adding semantic explanations of Steam tags helps a neural recommender understand games better.

We compared three text representations:

| Condition | Text Representation |
|---|---|
| **A** | Game title only |
| **B** | Game title + Steam tag list |
| **C** | Game title + LLM-generated tag descriptions |

The model architecture, training setup, evaluation impressions, negatives, and hyperparameters were kept controlled across the three conditions.

---

## Main Idea

Steam tags such as `Souls-like`, `Roguelike`, `Metroidvania`, and `Colony Sim` contain important gameplay information. However, the tag name alone may not fully describe what kind of experience the game provides.

So, we generated short tag descriptions using an LLM.

Example:

```text
Souls-like → punishing combat, stamina management, dying repeatedly, recovering souls, deliberate exploration
Roguelike → procedural runs, permadeath, randomized loot, escalating difficulty, build experimentation
Metroidvania → interconnected map, locked areas, ability-gated exploration, backtracking, atmospheric world
```

These descriptions were added to the game representation and passed into the recommendation model.

---

## Contributions

- Built a leakage-safe Steam dataset using iterative sparsity filtering and strict temporal splits.
- Implemented NRMS for the game recommendation domain.
- Tested three controlled text conditions: title only, title with tags, and title with LLM-generated tag descriptions.
- Evaluated the model on 3,000 held-out test impressions.
- Showed that LLM descriptions can help, but only when they are short, frequency-ordered, and kept close to the original tag keywords.
- Turned a failed direct replication into a useful design lesson about prompt design, token budget, and feature ordering.

---

## Dataset

The dataset was created from Steam game metadata and user interaction data.

After filtering, the final dense dataset contains:

| Statistic | Value |
|---|---:|
| Users | 15,660 |
| Games | 3,825 |
| Total interactions | 174,144 |
| Unique tags | 433 |
| Test impressions | 3,000 |
| Train samples | 519,002 |

The dataset uses strict temporal splitting, meaning the model is trained on earlier interactions and evaluated on later interactions. This makes the evaluation more realistic and reduces leakage.

---

## Model Architecture

The project uses an **NRMS-style neural recommender**.

The system includes:

- A game encoder based on frozen `DistilBERT`
- Additive attention pooling
- A user encoder with multi-head self-attention
- Dot-product scoring between user vectors and candidate game vectors

The model learns a representation of each user from their previous game interactions and ranks candidate games based on relevance.

---

## Text Conditions

### Condition A — Title Only

```text
Game Title
```

### Condition B — Title + Tags

```text
Game Title. Tags: Action, RPG, Souls-like, Singleplayer
```

### Condition C — Title + LLM Tag Descriptions

```text
Game Title.
Souls-like: punishing combat, stamina management, dying repeatedly.
RPG: character progression, quests, skill development.
Action: fast combat, reflex-based gameplay, movement.
```

Condition C gives the model more semantic information about what the tags mean in gameplay terms.

---

## Results

| Condition | AUC | MRR | nDCG@5 | nDCG@10 | Hit@1 | Hit@5 | Hit@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A: Title only** | 0.630 | 0.298 | 0.278 | 0.347 | 0.156 | 0.425 | 0.640 |
| **B: Title + tags** | 0.700 | **0.345** | **0.336** | **0.411** | **0.181** | **0.510** | 0.742 |
| **C: Title + LLM descriptions** | **0.703** | 0.324 | 0.317 | 0.399 | 0.156 | 0.499 | **0.753** |

---

## Key Finding

Condition C achieved the highest AUC, showing that LLM-generated descriptions can improve overall ranking quality.

However, the improvement over tags alone was small. This means that Steam tags already contain strong recommendation signal, and LLM descriptions only help when they are carefully designed.

---

## What We Learned

A direct copy of the original LLM-description method did not work well for Steam.

Long descriptions caused useful information to be truncated by the encoder token limit. Also, generic tags such as `Singleplayer` and `Action` often appeared first and dominated the input.

The final improvement came from three design choices:

1. Keeping descriptions short
2. Ordering tags by frequency so rare and more meaningful tags appear first
3. Keeping tag keywords close to their generated descriptions

This shows that LLM-generated text is useful, but only when it is compact, controlled, and aligned with the model’s input budget.

---

## Repository Structure

```text
Steam-Game-Recommendation/
│
├── data_preparation.ipynb
├── nrms.ipynb
├── steam_prepared_data.zip
├── nrms_output.zip
├── assets/
│   └── pacman.gif
└── README.md
```

---

## File Descriptions

| File | Description |
|---|---|
| `data_preparation.ipynb` | Prepares the Steam dataset, cleans interactions, applies sparsity filtering, creates temporal splits, and exports model-ready files. |
| `nrms.ipynb` | Runs the NRMS model, builds the three text conditions, trains models, and evaluates results. |
| `steam_prepared_data.zip` | Prepared dataset files used for training and evaluation. |
| `nrms_output.zip` | Output files including generated tag descriptions and final results. |
| `assets/pacman.gif` | GIF used in the README header. |

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/AseelIsCoding/Steam-Game-Recommendation.git
cd Steam-Game-Recommendation
```

### 2. Open the Notebooks

This project is designed to run in Google Colab.

Run the notebooks in this order:

```text
1. data_preparation.ipynb
2. nrms.ipynb
```

### 3. Prepare the Data

If you already have:

```text
steam_prepared_data.zip
```

you can upload it directly when running the model notebook.

Otherwise, run:

```text
data_preparation.ipynb
```

to generate the prepared dataset.

### 4. Train and Evaluate the Model

Run:

```text
nrms.ipynb
```

The notebook trains and evaluates the model under the three conditions:

```text
A: Title only
B: Title + tag list
C: Title + LLM-generated tag descriptions
```

Outputs are saved into:

```text
nrms_output/
```

---

## Technologies Used

- Python
- PyTorch
- Transformers
- DistilBERT
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- DeepSeek API
- Google Colab

---

## Project Highlights

- Neural recommender system for Steam games
- NRMS implementation
- LLM-generated feature augmentation
- Steam tag understanding
- Leakage-safe temporal evaluation
- AUC, MRR, nDCG, and Hit@k metrics
- Practical analysis of token budget and prompt design

---

## Authors

- **Aseel A. Almehmadi**
- **Hadeel A. Sirdar**
- **Fatima A. Almaashi**

University of Jeddah  
College of Computer Science and Engineering  
Department of Computer Science and Artificial Intelligence  

---

## Acknowledgment

This project was completed as part of the **CCAI 422 Recommender Systems** course at the University of Jeddah.

The project is inspired by the idea of enhancing recommendation models using LLM-generated category descriptions and extends this idea from news recommendation to Steam game recommendation.

---

## License

This repository is for academic and educational use.
