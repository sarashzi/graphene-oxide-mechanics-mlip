# Graphene Oxide Mechanics with Machine-Learning Interatomic Potentials

This repository contains the computational workflow and scripts associated with the study:

**“Oxidation-dependent mechanical response of graphene oxide: Improving the reliability of atomistic modelling by ab initio machine learning simulations”**

S. Shahbazi Fashtali, P. M. Piaggi, and G. Zollo  
*Physical Review Materials*

## Overview

This work investigates the mechanical response of graphene oxide (GO) using a machine-learning interatomic potential based on MACE and density-functional theory (DFT) reference calculations.

The computational workflow consists of:

1. Molecular dynamics simulations using a MACE foundation model.
2. Extraction of representative atomic configurations from MD trajectories.
3. DFT calculations with Quantum ESPRESSO to calculate reference energies and forces.
4. Evaluation and selection of the MACE foundation model.
5. Preparation of training and validation datasets.
6. Fine-tuning of the selected MACE model on the DFT reference data.
7. Iterative extension of the training dataset with additional configurations (active learning).
8. Molecular dynamics and uniaxial tensile simulations using the fine-tuned potential to calculate the mechanical properties of GO.

## Repository structure

```text
01_foundation_model_md/
    md_foundation.py
    job_foundation.sh

02_dft_dataset_generation/
    extract_snapshots_qe_npt.ipynb
    extract_snapshots_qe_nvt.ipynb

03_dataset_preparation/
    data_preparation.py

04_mace_finetuning/
    job_finetune.sh

05_production_md/
    md_finetuned.py
    job_finetuned.sh

models/
    mace_go_pbe_finetuned.model

examples/
    go_20_oxidation_OH_O_1.data
    go_larger_system_20_oxidation_OH_O_1.data



```markdown
## Example structures

Two example graphene oxide structures with 20% oxidation and OH/O = 1 are provided:

- `go_20_oxidation_OH_O_1.data`: example structure for the foundation-model MD and DFT dataset-generation workflow.
- `Go_larger_system_20_oxidation_OH_O_1.data`: larger example structure for production MD and uniaxial tensile simulations using the fine-tuned MACE potential.

## Fine-tuned model

The `models/` directory contains the fine-tuned MACE potential used for the production molecular dynamics simulations. The model was fine-tuned on PBE DFT energies and atomic forces for graphene oxide configurations containing C, O, and H.
