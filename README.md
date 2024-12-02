# Enhancement of Hash Functions Employing Different Approaches

## General Review

We tested three separate optimization methods: Hill Climbing, Simulated Annealing, and Nelder-Mead. We applied them in this work to maximize hash table parameters. Here, the objective is to reduce hash collisions in the hash function `h(x) = (ax + b) mod m`. The parameters we wish to optimize are `a`, `b`, and `m`.

The hash function with linear probing helps us to solve collisions. When the load factor rises above 0.75, we scale. This article compares the three optimization methods to get the ideal values to lower hash collisions.

## Setup & Configuration

### Required Conditions

Pandas, NumPy, Python 3.8 or newer, and Matplotlib

### Overseeing the Project

Perform the primary script:

```bash
python CSCI270_F24_PA5_AlyamanMassarani_120124_hashopt.py
```

## Review of Findings

The three algorithms show amazing trends in the results of the script:

### Comparison Between Methods

Simulated Annealing showed the best performance, achieving the best results in 47 out of 50 test iterations. Meanwhile, Hill Climbing performed best in 3 iterations, while Nelder-Mead did not achieve the best performance in any iteration

Since the provided plots by the code show that it works rather better on average than other approaches, simulated Annealing is the most reliable solution for this problem.

## Final Finish

In this sense, simulated Annealing is clearly the best approach in this comparison for this particular hash function optimization difficulty. Most likely this can be explained by:

1. Can avoid local optima with its temperature-based acceptance method.
2. Better exploration of the solution space while still achieving good results in the end

Finally, among the three methods tested, simulated Annealing seems to be the best one.

Contact: massaraniah@g.cofc.edu.
