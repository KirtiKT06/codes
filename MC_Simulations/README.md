# Monte Carlo Sampling Methods

This repository contains Python implementations of Monte Carlo methods
for generating non-uniform random numbers, with a focus on:

- Inverse transform sampling
- Acceptance–rejection sampling
- Metropolis sampling 

The code is written in a modular and object-oriented manner and is intended
for educational and coursework purposes.

---

## Implemented Distributions

- Linear probability density function  
  f(x) = 2x, x $\in$ $[0,1]$
- Symmetric triangular distribution on $[-1,1]$
- Gaussian distribution

Each distribution provides:
- analytical probability density
- inverse cumulative distribution function (where applicable)
- theoretical mean and variance

---

## Sampling Methods

- **Inverse Transform Sampling**
- **Acceptance–Rejection Sampling**
- **Metropolis Sampling**

The samplers are implemented generically and can be applied to any
distribution that satisfies the required interface.

---

## Problems 

The LJ (12-6) potential has been calculated for a system of 
256 particles taking into account different values of $r_c$
and finding the most fitting value.

---

## How to Run

Clone the repository and run the main script:

```bash
python main.py
