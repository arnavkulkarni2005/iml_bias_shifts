# Bias Shifts in LLM Representations

**A Deep Investigation into the "Fairness Facade" of Large Language Models**

---

## Overview

This research project investigates a critical question in AI safety and ethics: Do alignment techniques actually remove social biases from Large Language Models, or do they merely hide them beneath a superficial layer of "correct" responses?

By analyzing the internal activations of the Phi-3-mini model and utilizing advanced techniques like linear probing and geometric analysis, this study reveals evidence of a **"Fairness Facade"** — a phenomenon where models appear unbiased in their outputs while maintaining intact biased representations in their internal structure.

### The Central Hypothesis

When an LLM is given a "progressive" or "anti-racist" system prompt, it learns to generate politically correct responses. However, the model's internal representation space may continue to encode the very biases it claims to reject. This creates a dangerous disconnect between what the model appears to believe and what it actually "knows" internally.

---

## Project Structure

```
iml_bias_shifts/
├── extended_experiments/          # Core research scripts
│   ├── experiments_utils.py       # Utilities for model handling
│   ├── rq1_geometry.py           # Gender subspace analysis
│   ├── rq2_intervention.py       # Alignment technique comparison
│   ├── rq3_utility_tax.py        # Confidence degradation analysis
│   ├── rq4_cot_leak.py           # Chain-of-thought probing
│   └── rq5_cross_attribute.py    # Racial bias generalization
├── requirements.txt              # Python dependencies
├── rq1_geometry.png             # Visualization of gender clusters
└── rq*_results.txt              # Detailed experimental outputs
```

### Core Components

**experiments_utils.py**
- Model initialization (Phi-3-mini) with device management
- Embedding extraction at specific transformer layers
- Dataset management for gendered/racialized names
- Probe training utilities for binary classification

**Research Question Scripts (rq1-rq5)**
- Each script corresponds to one research question
- Self-contained experiments with clear outputs
- Reproducible with consistent random seeds

---

## Research Questions & Key Findings

### RQ1: Is There a Gender Subspace?

**Question:** Does the model's internal representation space naturally cluster by gender, even without explicit gender markers in the input?

**Method:**
- Extract embeddings for 20 common male names and 20 common female names
- Use Principal Component Analysis (PCA) to identify the dominant axes of variation
- Visualize the separation in the reduced-dimensional space

**Finding:**
> **The gender attribute is extraordinarily dominant in the model's latent space.**

- The first Principal Component (PC1) alone explains **16.76%** of the total variance
- In a high-dimensional space with thousands of features, having a single attribute account for nearly 17% of variance indicates extreme segregation
- The visualization shows **clear linear separation** between male and female name clusters

**Interpretation:**
The model has learned to organize information along a "gender axis" that is so prominent it's one of the first patterns PCA discovers. This isn't accidental — it means gender is a fundamental organizing principle in how the model represents the world.

**Visualization:** `rq1_geometry.png` shows the stark clustering in PCA space, with male and female names forming distinct, well-separated clouds.

---

### RQ2: Prompting vs. Mathematical Intervention

**Question:** Can we remove gender bias by asking nicely, or do we need to surgically alter the model's weights?

**Method:**
Three conditions tested:
1. **Baseline:** No special instructions
2. **Soft Alignment (Progressive Prompt):** System message requesting gender-neutral, progressive responses
3. **Hard Alignment (Linear Erasure):** Mathematical removal of the gender subspace using orthogonal projection

A linear probe (binary classifier) is trained to predict gender from internal activations. If the probe can still accurately classify, the gender information remains accessible.

**Findings:**

| Condition | Probe Accuracy | Interpretation |
|-----------|---------------|----------------|
| Baseline | 1.00 (100%) | Gender is perfectly recoverable |
| Soft Alignment | 1.00 (100%) | **Prompt changes nothing internally** |
| Hard Alignment | 0.94 (94%) | Mathematical erasure partially works |

**Critical Insight:**
> **Prompting is cosmetic. The model's internal representation of gender remains completely intact even when instructed to be progressive.**

The progressive prompt affects only the model's **output behavior**, not its **internal knowledge structure**. This is the core of the Fairness Facade: the model acts fair while "thinking" in biased categories.

Even hard alignment doesn't fully eliminate gender classification (94% is still very high), suggesting gender is deeply embedded across multiple representational dimensions.

---

### RQ3: The Alignment Tax

**Question:** When forced into a "progressive persona," does the model pay a price in confidence when stating biological facts?

**Method:**
- Present the model with objective biological questions (e.g., "Which biological sex typically produces sperm?")
- Measure two metrics:
  - **Accuracy:** Does it get the answer right?
  - **Confidence:** How certain is the model (via Negative Log-Likelihood)?
- Compare baseline vs. progressive persona

**Findings:**

| Metric | Baseline | Progressive | Change |
|--------|----------|-------------|--------|
| Accuracy | 100% | 100% | No change |
| Avg. Confidence Margin | 0.1802 | 0.1145 | **-36.5% drop** |

**Interpretation:**
The progressive persona creates internal conflict. While the model still arrives at the correct factual answer, it does so with significantly reduced confidence. 

This suggests:
- The model "knows" the biological facts
- The progressive framing creates cognitive dissonance
- The model becomes less certain when forced to reconcile progressive values with biological reality

**Metaphor:** Imagine being asked "2+2=?" but first being told "Numbers are a social construct." You'd still answer "4," but with less confidence because of the confusing context.

---

### RQ4: Chain-of-Thought Leakage

**Question:** When the model "thinks out loud" using step-by-step reasoning, can we catch it using gender information even when the final answer is neutral?

**Method:**
- Prompt: "Is [Name] more likely to be a doctor or a nurse? Let's think step by step."
- Extract activations from the **internal reasoning tokens** (the hidden "thought" process)
- Train a probe on these intermediate activations
- Compare to the final output (which may be carefully gender-neutral)

**Findings:**
> **Reasoning Token Probe Accuracy: 1.00 (100%)**

**Critical Insight:**
Even when the model produces a perfectly neutral, unbiased final answer, its **internal reasoning process** immediately identifies and uses the subject's gender. The "thinking" is biased even when the conclusion is fair.

**Analogy:**
It's like a hiring manager who internally categorizes candidates by gender (consciously or unconsciously) during deliberation, but then delivers a fair-sounding justification for their decision. The bias exists in the cognitive process, not just the outcome.

This reveals a profound limitation of evaluating AI systems solely by their outputs. The internal processing may be fundamentally biased even when surface-level responses appear equitable.

---

### RQ5: Does This Generalize to Race?

**Question:** Is the Fairness Facade specific to gender, or does it extend to other protected attributes like race?

**Method:**
- Use European American vs. African American names (validated by sociological research)
- Measure sentiment association using a lexicon of positive/negative words
- Compare:
  - **Baseline:** No special instructions
  - **Anti-Racist Prompt:** Explicit request for equitable, anti-racist responses
- Train probes to classify race from internal activations

**Findings:**

**Surface-Level Bias (Sentiment Association):**
- The anti-racist prompt **reduces** the observable bias gap
- The model generates more equitable sentiment associations when prompted

**Internal Representation (Probe Accuracy):**
> **Probe Accuracy: 1.00 (100%) in both baseline and anti-racist conditions**

**Interpretation:**
The Fairness Facade extends to racial bias:
1. **Externally:** The anti-racist prompt improves output fairness
2. **Internally:** Racial categorization remains perfectly intact at the representation level

This confirms the hypothesis that alignment techniques work primarily on **output filtering**, not on **internal representation restructuring**.

**Broader Implication:**
If this pattern holds across attributes (gender, race, age, religion, etc.), it suggests that current alignment methods may be fundamentally limited. They teach models to "say the right thing" without changing how they fundamentally encode social categories.

---

## Setup and Installation

### System Requirements

- **Python:** 3.11 or higher recommended
- **Hardware:** GPU with CUDA support strongly recommended
  - The Phi-3-mini model requires ~8GB VRAM for inference
  - CPU-only mode will work but be significantly slower
- **Operating System:** Linux, macOS, or Windows with WSL

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/arnavkulkarni2005/iml_bias_shifts.git
cd iml_bias_shifts
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch` — PyTorch for model execution
- `transformers` — Hugging Face library for loading Phi-3-mini
- `scikit-learn` — PCA, linear probes, and evaluation metrics
- `numpy` — Numerical operations
- `matplotlib` — Visualization generation

3. **Verify CUDA (optional but recommended):**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `True`, you're ready for GPU acceleration. If `False`, the scripts will fall back to CPU (slower but functional).

---

## Usage

### Running Individual Experiments

Each research question has its own standalone script:

```bash
# RQ1: Analyze gender geometry
python extended_experiments/rq1_geometry.py

# RQ2: Compare alignment interventions
python extended_experiments/rq2_intervention.py

# RQ3: Measure the alignment tax
python extended_experiments/rq3_utility_tax.py

# RQ4: Probe chain-of-thought reasoning
python extended_experiments/rq4_cot_leak.py

# RQ5: Test cross-attribute generalization
python extended_experiments/rq5_cross_attribute.py
```

### Expected Runtime

- **RQ1:** ~5-10 minutes (includes PCA and visualization)
- **RQ2:** ~10-15 minutes (three model configurations)
- **RQ3:** ~5-7 minutes (confidence comparison)
- **RQ4:** ~8-12 minutes (reasoning token extraction)
- **RQ5:** ~10-15 minutes (sentiment + probe training)

*Times assume GPU acceleration. CPU-only may be 3-5x slower.*

### Output Files

Each script generates:
- **`rq*_results.txt`** — Detailed numerical results and statistics
- **`rq1_geometry.png`** — PCA visualization (RQ1 only)

Results are logged to the terminal and saved to files for later analysis.

---

## Interpreting the Results

### What Does 100% Probe Accuracy Mean?

A linear probe with **100% accuracy** means:
- A simple straight line (or hyperplane) can perfectly separate the two classes in the representation space
- The attribute (gender, race) is **linearly encoded** and easily extractable
- The model hasn't "forgotten" or "unlearned" the attribute

**Why This Matters:**
If biases were truly removed, probe accuracy would drop to ~50% (random chance). High accuracy indicates the information remains accessible, even if the model's outputs don't reflect it.

### What Is "Confidence Margin"?

In RQ3, we measure:
```
Confidence = probability(correct answer) - probability(incorrect answer)
```

A **high margin** (e.g., 0.95 vs 0.05 = 0.90) means the model is very certain.
A **low margin** (e.g., 0.60 vs 0.40 = 0.20) means the model is hesitant.

The 36.5% drop in confidence margin shows the progressive persona creates internal uncertainty, even when the answer remains correct.

### What Does PCA Tell Us?

**Principal Component Analysis (PCA)** finds the directions of maximum variance in high-dimensional data.

- **PC1 explains 16.76%:** Gender is the single most important organizing principle
- **Clear clustering:** Names don't just slightly correlate with gender — they form distinct groups

This isn't about individual word associations. It's about the fundamental geometry of the model's knowledge representation.

---

## Deep Dive: The Fairness Facade Explained

### Why This Matters for AI Safety

Current AI alignment techniques focus heavily on **Reinforcement Learning from Human Feedback (RLHF)** and prompt engineering. These methods teach models:
- "When asked about gender, give neutral responses"
- "Don't make assumptions based on protected attributes"
- "Decline inappropriate requests"

**The Problem:** These techniques work at the **output layer**, not the **representation layer**.

### The Two-Layer Model of Bias

```
[Internal Representation]  →  [Output Generation]
   (Biased structure)          (Filtered response)
```

**Current alignment** modifies the right side (output) without changing the left side (internal knowledge).

**Consequence:** The model:
- Can generate fair-sounding text
- Can pass bias benchmarks
- Can avoid obvious stereotypes in conversation
- **But** still organizes information using biased categories internally

### Real-World Risks

1. **Adversarial Extraction:** Sophisticated users might find ways to probe these hidden representations through carefully crafted prompts

2. **Reasoning Failures:** If the model "thinks" in biased categories during chain-of-thought, complex reasoning tasks might inherit these biases even when simple Q&A doesn't

3. **Fine-Tuning Vulnerabilities:** Fine-tuning on downstream tasks might "unlock" the latent biases that were merely suppressed, not removed

4. **False Confidence:** Deployers might believe biases are solved when they're only hidden, leading to insufficient monitoring

---

## Key Takeaways

1. **Prompting ≠ Debiasing**
   - System prompts change behavior, not internal structure
   - 100% probe accuracy persists even with "progressive" personas

2. **Alignment Creates Cognitive Dissonance**
   - Models forced into progressive personas show reduced confidence
   - The "alignment tax" suggests internal conflict between training data and alignment instructions

3. **Chain-of-Thought Leaks Internal State**
   - Even when final answers are neutral, reasoning steps reveal biased categorization
   - Evaluating only outputs misses the biased process

4. **The Fairness Facade Is Generalizable**
   - Gender bias patterns replicate for racial bias
   - Suggests a fundamental limitation of current alignment methods

5. **We Need Deeper Interventions**
   - Surface-level alignment creates a false sense of safety
   - True debiasing requires representational restructuring, not just output filtering

---

## Future Research Directions

1. **Scaling to Larger Models:** Does the Fairness Facade persist in GPT-4, Claude, or Gemini scale models?

2. **Other Attributes:** Age, religion, disability, socioeconomic status — how widespread is this phenomenon?

3. **Mechanistic Interpretability:** Can we identify the specific attention heads or MLP layers responsible for encoding biased attributes?

4. **Robust Alignment:** What techniques actually modify internal representations rather than just outputs?

5. **Evaluation Frameworks:** How should we test for the Fairness Facade in production systems?

---

## Theoretical Background

### What Are Linear Probes?

A **linear probe** is a simple classifier (typically logistic regression or a single-layer neural network) trained to predict a property from a model's internal activations.

**Why use them?**
- If a linear probe succeeds, the information is **linearly accessible** in the representation
- Linear accessibility suggests the model is explicitly encoding the attribute, not just implicitly correlating with it
- High probe accuracy = the information is "close to the surface" and easily extractable

### What Is Geometric Analysis?

Modern LLMs represent concepts as points in high-dimensional space (often 768 to 4096 dimensions for each token).

**Key insight:** The **geometry** of this space reveals conceptual structure:
- **Distance:** Similar concepts are closer together
- **Direction:** Attributes form consistent vector directions (e.g., "gender vector")
- **Clustering:** Related concepts form dense regions

By using PCA to reduce dimensions, we can visualize this geometry and see if social categories create structured partitions.

### What Is "Activation Space"?

At each layer of a transformer, tokens have an associated vector (their **activation** or **hidden state**).

This research examines **Layer 16** of Phi-3-mini, which is roughly mid-way through the model. At this depth:
- Early layers: Basic syntax and surface-level features
- Middle layers: Semantic meaning and abstract concepts ← **We probe here**
- Late layers: Task-specific reasoning and output preparation

Layer 16 captures where the model has formed stable semantic representations but hasn't yet specialized for output generation.

---

## Contributing

This repository is open for community contributions:

- **Replication Studies:** Test these methods on different models (Llama, Mistral, GPT variants)
- **Extended Analyses:** Add new research questions or attributes
- **Visualization:** Improve plots and interpretability
- **Documentation:** Clarify methodology or add tutorials

For major changes, please open an issue first to discuss your approach.

---

## Citation

If you use this work in your research, please cite:

```
@misc{iml_bias_shifts_2024,
  author = {arnavkulkarni2005},
  title = {Bias Shifts in LLM Representations: Investigating the Fairness Facade},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/arnavkulkarni2005/iml_bias_shifts}
}
```

---

## License

*No license information currently specified in the repository.*

---

## Acknowledgments

- **Phi-3-mini:** Microsoft's efficient small language model used for all experiments
- **Hugging Face Transformers:** Infrastructure for model loading and inference
- **scikit-learn:** Machine learning utilities for probing and analysis

---

## Contact

For questions, discussions, or collaboration opportunities:
- **GitHub Issues:** [github.com/arnavkulkarni2005/iml_bias_shifts/issues](https://github.com/arnavkulkarni2005/iml_bias_shifts/issues)
- **Repository Owner:** [@arnavkulkarni2005](https://github.com/arnavkulkarni2005)

---

## Related Work

This research builds on several key areas in AI safety and interpretability:

1. **Representation Analysis:** Work by Anthropic, OpenAI, and academic labs on understanding neural network internals

2. **Bias in NLP:** Studies showing that word embeddings (Word2Vec, GloVe) encode social biases — this extends those findings to modern LLMs

3. **Alignment Research:** RLHF, Constitutional AI, and other techniques that attempt to align model behavior with human values

4. **Mechanistic Interpretability:** Circuit analysis, causal tracing, and other methods for understanding how models implement specific behaviors

---

**Last Updated:** February 2026  
**Repository:** [github.com/arnavkulkarni2005/iml_bias_shifts](https://github.com/arnavkulkarni2005/iml_bias_shifts)
