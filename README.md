Mars Rover Tracking with Hidden Markov Models (ECE368)
 
An exact probabilistic inference project tracking the trajectory of a wandering Mars Rover across a 2D grid using noisy and missing sensor observations.

**Overview**
A rover wanders on a 12x8 grid. Its exact coordinates and previous actions are hidden; ground control only receives noisy, discretized sensor readings subject to telemetry loss (missing observations).This project implements exact Hidden Markov Model (HMM) algorithms to reconstruct the rover's trajectory:

<img width="345" height="220" alt="image" src="https://github.com/user-attachments/assets/44f003d6-1f92-46e2-8c95-885599537d47" />

**Algorithms Used**

1. Forward-Backward Algorithm: Calculates smoothed posterior marginal distributions across all time steps.
2. Viterbi Algorithm: Computes the globally most likely state sequence (Maximum A Posteriori / MAP sequence) 
3. Missing Data Imputation: Formulates sensor loss ($None$) in message passing without breaking normalization or Markov properties.

**Rover Demo**

<img width="955" height="305" alt="image" src="https://github.com/user-attachments/assets/6b20f99b-a61a-4cea-adf2-f39faa947adf" />


