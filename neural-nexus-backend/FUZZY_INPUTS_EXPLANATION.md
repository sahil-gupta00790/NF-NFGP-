# Fuzzy System Inputs - Simple Explanation

## What Are the 3 Inputs?

The **3 inputs** to the fuzzy system are **statistical features** extracted from the neural network's output.

## How It Works

### Step 1: Get Model Output
When the neural network processes an input (e.g., an image), it produces an output (e.g., predictions for 10 classes).

### Step 2: Extract Statistics
From that output, we calculate **statistical features** that describe the model's behavior:

1. **Mean** - Average value of all outputs
2. **Standard Deviation** - How spread out the outputs are
3. **Minimum** - Smallest output value
4. **Maximum** - Largest output value  
5. **Median** - Middle value of all outputs

### Step 3: Use First 3 Features
Since the fuzzy config specifies `num_inputs = 3`, we take the **first 3 features**:
- **Input 1: Mean** - Average output value
- **Input 2: Standard Deviation** - Output spread/variance
- **Input 3: Minimum** - Smallest output value

## In Simple Terms

Think of it like this:

```
Neural Network Output: [0.1, 0.8, 0.2, 0.9, 0.3, 0.7, 0.1, 0.6, 0.4, 0.5]
                         ↓
                    Calculate Statistics
                         ↓
    Mean: 0.46  ─────────┐
    Std:  0.28  ─────────┤──→ Fuzzy Input 1: Mean (0.46)
    Min:  0.10  ─────────┤──→ Fuzzy Input 2: Std Dev (0.28)
    Max:  0.90           │──→ Fuzzy Input 3: Min (0.10)
    Median: 0.45         │
                         ↓
                    Normalize to [0, 1]
                         ↓
              [0.46, 0.28, 0.10] → Fuzzy System
```

## What Do These Inputs Tell Us?

1. **Mean (Input 1)**: 
   - **High** = Model is generally confident
   - **Low** = Model is generally uncertain

2. **Standard Deviation (Input 2)**:
   - **High** = Model has mixed confidence (some outputs high, some low)
   - **Low** = Model is consistent (all outputs similar)

3. **Minimum (Input 3)**:
   - **High** = Even the worst output is decent
   - **Low** = Model has some very low-confidence outputs

## Why These Features?

These statistics capture **behavioral characteristics** of the model:
- How confident it is overall (mean)
- How consistent its predictions are (std dev)
- How bad its worst case is (minimum)

The fuzzy system uses these behavioral features to evaluate the model's "fitness" from a different perspective than just accuracy.

## Example

**Scenario 1: Confident Model**
```
Output: [0.05, 0.90, 0.02, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00]
Mean: 0.10 (low - but that's because most are near zero)
Std: 0.28 (high - very spread out)
Min: 0.00 (very low)
→ Fuzzy system sees: "Model is very confident in one class, uncertain in others"
```

**Scenario 2: Uncertain Model**
```
Output: [0.10, 0.11, 0.09, 0.12, 0.10, 0.11, 0.09, 0.10, 0.10, 0.08]
Mean: 0.10 (low)
Std: 0.01 (very low - all similar)
Min: 0.08 (relatively high)
→ Fuzzy system sees: "Model is evenly uncertain across all classes"
```

## Summary

The **3 inputs** are:
1. **Mean** of model outputs (average confidence)
2. **Standard Deviation** of model outputs (consistency)
3. **Minimum** of model outputs (worst-case confidence)

These are **behavioral features** that describe how the model behaves, not just what it predicts. The fuzzy system uses these to provide a complementary fitness score alongside the neural network's accuracy.

