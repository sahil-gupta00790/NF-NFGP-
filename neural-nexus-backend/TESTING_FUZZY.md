# Testing Hybrid Neuro-Fuzzy Co-Evolution

## Quick Start

### 1. Run Unit Tests

```bash
cd neural-nexus-backend
python test_fuzzy_evolution.py
```

This will:
- Test the fuzzy inference system
- Test helper functions
- Generate example config files in `test_configs/`

### 2. Test Backward Compatibility (NN-Only)

**Config:** `test_configs/config_nn_only.json`

```json
{
  "model_class": "SimpleCNN",
  "generations": 3,
  "population_size": 5,
  "use_fuzzy": false,  // or omit this field
  ...
}
```

**What to verify:**
- System behaves identically to original implementation
- Logs show: `"Fuzzy co-evolution disabled (NN-only)"`
- No errors related to fuzzy parameters
- Evolution completes successfully

### 3. Test Hybrid Neuro-Fuzzy Evolution

**Config:** `test_configs/config_hybrid.json`

```json
{
  "model_class": "SimpleCNN",
  "generations": 3,
  "population_size": 5,
  "use_fuzzy": true,
  "fuzzy_config": {
    "num_inputs": 5,
    "num_rules": 5,
    "membership_per_input": 3,
    "alpha": 0.7
  },
  ...
}
```

**What to verify:**
- Logs show: `"Fuzzy co-evolution enabled"`
- Logs show: `"Initialized fuzzy genome with N parameters"`
- Fitness combines NN and fuzzy components
- Final result includes `best_fuzzy_parameters`
- Evolution completes successfully

## Testing via API

### Using curl

```bash
# Start backend services first (docker-compose up)

# Test NN-only
curl -X POST "http://localhost:8000/api/evolver/start" \
  -F "model_definition=@test_model.py" \
  -F "config_json=$(cat test_configs/config_nn_only.json)" \
  -F "use_standard_eval=true"

# Test Hybrid
curl -X POST "http://localhost:8000/api/evolver/start" \
  -F "model_definition=@test_model.py" \
  -F "config_json=$(cat test_configs/config_hybrid.json)" \
  -F "use_standard_eval=true"
```

### Check Task Status

```bash
# Replace TASK_ID with the ID returned from /start
curl "http://localhost:8000/api/evolver/status/TASK_ID"
```

## Testing via Frontend

1. Start the frontend: `cd neural-nexus-frontend && npm run dev`
2. Navigate to the Evolver section
3. Upload model definition file
4. Copy config JSON from `test_configs/config_hybrid.json`
5. Paste into the config field
6. Select "Standard Evaluation"
7. Submit and monitor progress

## What to Check in Logs

### Successful NN-Only Evolution:
```
[Task XXX] Fuzzy co-evolution disabled (NN-only)
[Task XXX] Evolving 2 hyperparameters: ['num_filters', 'dropout_rate']
[Task XXX] Gen 1 Stats: MaxFit=XX.XX, AvgFit=XX.XX
...
[Task XXX] Evolution finished. Best Fitness: XX.XX
```

### Successful Hybrid Evolution:
```
[Task XXX] Fuzzy co-evolution enabled: {'num_inputs': 5, 'num_rules': 5, ...}
[Task XXX] Initialized fuzzy genome with 75 parameters
[Task XXX] Evaluating 5 individuals with 2 hyperparameters...
Fuzzy co-evolution enabled with 75 fuzzy parameters
...
[Task XXX] Best fuzzy parameters extracted (75 params)
[Task XXX] Evolution finished. Best Fitness: XX.XX
```

### Error Indicators (should NOT appear):
- `IndexError` related to chromosome slicing
- `ValueError: Chromosome too short`
- `ValueError: Weight Size mismatch`
- Any errors mentioning fuzzy parameter extraction

## Verification Checklist

- [ ] Unit tests pass (`python test_fuzzy_evolution.py`)
- [ ] NN-only evolution works (backward compatibility)
- [ ] Hybrid evolution works without errors
- [ ] Logs show correct fuzzy initialization
- [ ] Fitness values are finite and reasonable
- [ ] Best fuzzy parameters appear in final result (hybrid mode)
- [ ] Final model file is saved successfully
- [ ] No chromosome slicing errors in logs

## Troubleshooting

### Issue: "Fuzzy evaluation failed"
- Check that `eval_config` includes `input_shape` or `sample_input`
- Verify model output shape is compatible
- Check fuzzy_config `num_inputs` matches extracted features (default: 5)

### Issue: "Chromosome too short"
- Verify `num_fuzzy_params` is calculated correctly
- Check that chromosome initialization includes fuzzy genome when `use_fuzzy=True`

### Issue: "MF ordering violated"
- Check `mutate_fuzzy_params` preserves ordering
- Verify fuzzy parameter initialization creates valid MFs

## Example Model Definition

See `test_model.py` created by the test script for a simple CNN example.

