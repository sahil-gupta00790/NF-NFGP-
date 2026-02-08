# Fuzzy Genome Parameter Explanation

## Overview

The fuzzy genome encodes a complete Mamdani Fuzzy Inference System (FIS) with **56 parameters** for the default test configuration:
- `num_inputs = 3`
- `num_rules = 5`
- `membership_per_input = 3`
- `num_output_mfs = 3` (defaults to membership_per_input)

## Parameter Breakdown

### Total: 56 Parameters

```
56 = 27 (Input MFs) + 20 (Rules) + 9 (Output MFs)
```

---

## 1. Input Membership Functions (27 parameters)

**Formula:** `num_inputs × membership_per_input × 3`

**For our config:** `3 × 3 × 3 = 27 parameters`

### Structure
Each input variable has `membership_per_input` triangular membership functions (MFs).
Each triangular MF is defined by 3 parameters: `(a, b, c)` where:
- `a` = left vertex (where MF = 0)
- `b` = peak vertex (where MF = 1) 
- `c` = right vertex (where MF = 0)
- Constraint: `a ≤ b ≤ c`

### Layout
```
Input 0: MF0(a0, b0, c0), MF1(a1, b1, c1), MF2(a2, b2, c2)  → 9 params
Input 1: MF0(a0, b0, c0), MF1(a1, b1, c1), MF2(a2, b2, c2)  → 9 params
Input 2: MF0(a0, b0, c0), MF1(a1, b1, c1), MF2(a2, b2, c2)  → 9 params
                                                              ────────
                                                              27 params
```

### Example Values
```
Input 0, MF 0: a=0.0, b=0.0, c=0.1   (Low)
Input 0, MF 1: a=0.4, b=0.5, c=0.6   (Medium)
Input 0, MF 2: a=0.9, b=1.0, c=1.0   (High)
```

**Purpose:** These define how crisp input values are converted to fuzzy membership degrees.

---

## 2. Fuzzy Rules (20 parameters)

**Formula:** `num_rules × (num_inputs + 1)`

**For our config:** `5 × (3 + 1) = 20 parameters`

### Structure
Each rule has:
- **Antecedents:** `num_inputs` parameters (which MF to use for each input)
- **Consequent:** 1 parameter (which output MF to use)

### Layout
```
Rule 0: Input0_MF_idx, Input1_MF_idx, Input2_MF_idx, Output_MF_idx  → 4 params
Rule 1: Input0_MF_idx, Input1_MF_idx, Input2_MF_idx, Output_MF_idx  → 4 params
Rule 2: Input0_MF_idx, Input1_MF_idx, Input2_MF_idx, Output_MF_idx  → 4 params
Rule 3: Input0_MF_idx, Input1_MF_idx, Input2_MF_idx, Output_MF_idx  → 4 params
Rule 4: Input0_MF_idx, Input1_MF_idx, Input2_MF_idx, Output_MF_idx  → 4 params
                                                                      ────────
                                                                      20 params
```

### Example Rule
```
Rule 0: [1, 0, 2, 1]
Meaning:
  - IF Input0 is MF1 (Medium) 
    AND Input1 is MF0 (Low)
    AND Input2 is MF2 (High)
  - THEN Output is MF1 (Medium)
```

**Purpose:** These define the fuzzy logic rules that map input combinations to output.

---

## 3. Output Membership Functions (9 parameters)

**Formula:** `num_output_mfs × 3`

**For our config:** `3 × 3 = 9 parameters`

### Structure
Each output MF is a triangular function defined by `(a, b, c)` parameters.

### Layout
```
Output MF 0: a0, b0, c0  → 3 params (e.g., Low output)
Output MF 1: a1, b1, c1  → 3 params (e.g., Medium output)
Output MF 2: a2, b2, c2  → 3 params (e.g., High output)
                            ────────
                            9 params
```

### Example Values
```
Output MF 0: a=0.0, b=0.0, c=0.1   (Low fitness)
Output MF 1: a=0.4, b=0.5, c=0.6   (Medium fitness)
Output MF 2: a=0.9, b=1.0, c=1.0   (High fitness)
```

**Purpose:** These define the output fuzzy sets that are aggregated and defuzzified to produce the final fitness score.

---

## Complete Parameter Layout

```
Index Range    | Component              | Count | Description
───────────────┼────────────────────────┼───────┼─────────────────────────────
[0:27]         | Input MFs              | 27    | 3 inputs × 3 MFs × 3 params
[27:47]        | Rules                   | 20    | 5 rules × 4 params each
[47:56]        | Output MFs              | 9     | 3 output MFs × 3 params
───────────────┼────────────────────────┼───────┼─────────────────────────────
Total          |                         | 56    |
```

---

## How It Works

### 1. Fuzzification
Input crisp values (e.g., `[0.5, 0.3, 0.7]`) are converted to membership degrees using the **Input MFs** (27 params).

### 2. Rule Evaluation
Each rule (using **Rules** params, 20 params) computes its firing strength by taking the minimum (AND) of input memberships.

### 3. Aggregation & Defuzzification
Output MFs (9 params) are aggregated using max (OR) and then defuzzified using centroid method to produce final fitness [0, 1].

---

## Parameter Calculation for Different Configs

### Example 1: Current Test Config
```python
{
    "num_inputs": 3,
    "num_rules": 5,
    "membership_per_input": 3
}
```
**Total:** 27 + 20 + 9 = **56 parameters**

### Example 2: Larger Config
```python
{
    "num_inputs": 5,
    "num_rules": 5,
    "membership_per_input": 3
}
```
**Calculation:**
- Input MFs: 5 × 3 × 3 = 45
- Rules: 5 × (5 + 1) = 30
- Output MFs: 3 × 3 = 9
**Total:** 45 + 30 + 9 = **84 parameters**

### Example 3: Minimal Config
```python
{
    "num_inputs": 2,
    "num_rules": 3,
    "membership_per_input": 2
}
```
**Calculation:**
- Input MFs: 2 × 2 × 3 = 12
- Rules: 3 × (2 + 1) = 9
- Output MFs: 2 × 3 = 6
**Total:** 12 + 9 + 6 = **27 parameters**

---

## Key Constraints

1. **MF Ordering:** All triangular MFs must satisfy `a ≤ b ≤ c` (enforced during mutation)
2. **Rule Indices:** Antecedent/consequent indices are clamped to valid MF ranges
3. **Parameter Ranges:** MFs typically in [0, 1], rule indices in [0, num_MFs-1]

---

## Evolution

During evolution:
- All 56 parameters are mutated together as a contiguous block
- MF ordering constraints are preserved after mutation
- Rules participate in crossover (as a block)
- The entire fuzzy system co-evolves with neural network weights

---

## Visual Representation

```
Fuzzy Genome (56 parameters)
│
├─ Input MFs (27) ──────────────────────────────┐
│  │                                              │
│  ├─ Input 0: MF0, MF1, MF2 (9 params)         │
│  ├─ Input 1: MF0, MF1, MF2 (9 params)         │
│  └─ Input 2: MF0, MF1, MF2 (9 params)         │
│                                                 │
├─ Rules (20) ───────────────────────────────────┤
│  │                                              │
│  ├─ Rule 0: [inp0_MF, inp1_MF, inp2_MF, out_MF] │
│  ├─ Rule 1: [inp0_MF, inp1_MF, inp2_MF, out_MF] │
│  ├─ Rule 2: [inp0_MF, inp1_MF, inp2_MF, out_MF] │
│  ├─ Rule 3: [inp0_MF, inp1_MF, inp2_MF, out_MF] │
│  └─ Rule 4: [inp0_MF, inp1_MF, inp2_MF, out_MF] │
│                                                 │
└─ Output MFs (9) ──────────────────────────────┘
   │
   ├─ Output MF 0: (a, b, c) - 3 params
   ├─ Output MF 1: (a, b, c) - 3 params
   └─ Output MF 2: (a, b, c) - 3 params
```

---

## Summary

The **56 parameters** encode a complete fuzzy inference system:
- **27 params** define how inputs are fuzzified
- **20 params** define the fuzzy logic rules
- **9 params** define how outputs are defuzzified

Together, they create a fuzzy system that evaluates model behavioral features and produces a fitness score that complements the neural network's performance.

