# Monte Carlo Sampling Methods

This repository contains Python implementations of Monte Carlo methods
for generating non-uniform random numbers, with a focus on:

- Inverse transform sampling
- Acceptance–rejection sampling

The code is written in a modular and object-oriented manner and is intended
for educational and coursework purposes.

---

## Implemented Distributions

- Linear probability density function  
  f(x) = 2x, x \in [0,1]
- Symmetric triangular distribution on \([-1,1]\)
- Gaussian distribution
- LJ(12-6) potential calculation using r_cut

Each distribution provides:
- analytical probability density
- inverse cumulative distribution function (where applicable)
- theoretical mean and variance

---

## Sampling Methods

- **Inverse Transform Sampling**
- **Acceptance–Rejection Sampling**

The samplers are implemented generically and can be applied to any
distribution that satisfies the required interface.

---

## How to Run

Clone the repository and run the main script:

```bash
python main.py
