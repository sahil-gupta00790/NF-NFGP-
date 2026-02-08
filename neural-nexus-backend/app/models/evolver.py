# app/models/evolver.py
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class EvolverConfig(BaseModel):
    # --- REQUIRED ---
    model_class: str = Field(..., description="Name of the model class in the definition file.")
    generations: int = Field(..., gt=0, description="Number of generations to run.")
    population_size: int = Field(..., gt=1, description="Number of individuals in the population.")

    # --- NSGA-II & MULTI-OBJECTIVE ---
    nsga2_enabled: bool = Field(default=False, description="Enable Non-Dominated Sorting for multi-objective optimization.")
    # If False, the system defaults to single-objective (Accuracy).
    
    # --- NEURO-FUZZY CO-EVOLUTION ---
    use_fuzzy: bool = Field(default=False, description="Enable co-evolution of fuzzy logic behavioral rules.")
    num_fuzzy_params: int = Field(default=10, ge=0, description="Number of genes allocated to fuzzy membership functions/rules.")

    # --- HYPERPARAMETER EVOLUTION ---
    # This stores the ranges/types for parameters like 'learning_rate' or 'dropout'
    evolvable_hyperparams_config: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Schema defining which hyperparameters to evolve and their bounds."
    )

    # --- STANDARD GA SETTINGS ---
    selection_strategy: str = Field(default="tournament")
    crossover_operator: str = Field(default="one_point")
    mutation_operator: str = Field(default="gaussian")
    elitism_count: int = Field(default=1, ge=0)
    mutation_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    mutation_strength: float = Field(default=0.05)
    tournament_size: int = Field(default=3, gt=1)
    
    # --- MODEL ARGS ---
    model_args: Optional[List[Any]] = Field(default=None)
    model_kwargs: Optional[Dict[str, Any]] = Field(default=None)
    eval_config: Optional[Dict[str, Any]] = Field(default=None)

    model_config = {
        "extra": "ignore"
    }