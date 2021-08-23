# Note

## Overview

decode formulation
$$
\hat{W}=arg\max_{W}{P(O|W)P(W)}
$$
introduce phone $V$
$$
\hat{W}=arg\max_W\sum_{V}P(O|V,W)P(V|W)P(W) \\
\approx arg\max_W\{{\max_V{P_H(O|V)P_L(V|W)P_G(W)\}}}
$$
in HMM and language modelling
$$
\hat{w}_1^n=arg\max_{w_1^N}\{P(w_1^N)\cdot \sum_{s_1^T}P(o_1^T,s_1^T|w_1^N)\}\\
\approx arg\max_{w_1^n}\{P(w_1^n)\cdot\max_{s_1^T}P(o_1^T,s_1^T|w_1^N)\}\qquad\#Veterbi\ approximation\\
\approx arg\max_{w_1^N}\{P(w_1^N)\cdot\max_{s_1^T}\prod_{i=1}^TP(o_t,s_t|s_{t-1},w^N_1) \qquad \#HMM\ assumption\\
\approx arg\max_{w_1^n}\{P(w_1^N)\cdot\max_{s_1^T}\prod_{i=1}^T P(o_t|s_t,w_1^N)\cdot P(s_t|s_{t-1},w_1^N) \qquad\# HMM\ assumption
$$

implied constraint
$$
map(s_1^T) = w_1^N
$$


recombination principle
*Select the “best” among several paths in the network as soon as it appears that these paths have identically scored extensions, implying that the current best path will keep dominating the others.*

decode types

- network expansion
  - statical
  - dynamical
- search algorithm
  - time-synchronous
  - time-aysnchronous

main actions to be performed by any decoder

1. Generating hypothetical word sequences, usually by successive extensions.
2. Scoring the “active” hypotheses using the knowledge sources.
3. **Recombining** i.e. merging paths according to the knowledge sources.
4. **Pruning** to discard the most unpromising paths.
5. Creating “back-pointers” to retrieve the best sentence.

representation of the knowledge sources (optional)
four-level hierarchy

0. HMM states
1. stochastic m-gram LM
2. prefix-tree organization of the lexicon
3. context-dependent phonetic constraints

search-space coordinate system

1. the time index,
2. the LM state,
3. the phonetic arc,
4. the acoustic state. 



