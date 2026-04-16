# AGENTS.md

## Purpose
This file provides general guidelines for maintaining and modifying the repository. It ensures consistency, readability, and correctness of the codebase.

## Project Context
The repository implements a data science pipeline for financial modeling, including data processing, feature engineering, model training, evaluation, and reporting.

## Guidelines for Changes
- Maintain existing structure and logic unless a change is clearly justified.
- Prefer incremental improvements over large refactors.
- Do not overengineer solutions — prioritize clarity and practicality.
- Avoid introducing unnecessary abstractions or dependencies.

## Code Quality
- Keep code clear, readable, and modular.
- Check for inconsistencies, fix the syntax, keep everything aligned
- Keep existing functions focused and reasonably short.
- Do not add any new functions and variables

## Data & Pipeline Consistency
- Maintain alignment between data processing, feature engineering, and modeling steps.
- Preserve column naming conventions and data structures.
- Avoid breaking existing interfaces unless necessary.

## Reproducibility
- Ensure results are reproducible.
- Control randomness where applicable (e.g., random seeds).
- Validate outputs after changes.

## Modeling
- Follow existing modeling and evaluation approach unless changes are clearly justified.
- Ensure consistency between feature engineering, training, and evaluation steps.

## Limitations
- The pipeline is designed for demonstration purposes and does not reflect full production complexity.
- Results should not be interpreted as real-world financial or clinical outcomes.

## Validation
- Ensure the pipeline runs end-to-end after any modification.
- Show outputs after each modification.
- Verify that outputs remain consistent and logically correct.