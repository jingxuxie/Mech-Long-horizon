# Literature and novelty audit

Primary sources were inspected during this research session. This is not an exhaustive novelty clearance.

| Source | Established territory | Boundary for this project |
|---|---|---|
| Geiger et al., *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability*, JMLR 2025 | Executable causal abstraction and intervention alignment | Do not claim causal faithfulness or compositionality itself is new |
| Girard and Pappas, *Approximate Bisimulation*, 2011 | Simulation relations and approximate dynamical equivalence | The generic pair auditor and exact transition-closure induction are references, not novel algorithms |
| Miller, Chughtai, and Saunders, *Transformer Circuit Faithfulness Metrics are not Robust*, CoLM 2024 | Sensitivity to ablation and evaluation conventions | Explicitly fix persistent zero masks, protocols, initial states, and output risk |
| Hadad, Katz, and Bassan, *Formal Mechanistic Interpretability*, arXiv:2602.16823 | Formally guaranteed circuit discovery and non-monotonicity | Do not sell generic verification or signed-mask non-monotonicity as new |
| Zhang et al., *Dynamic Slicing for Deep Neural Networks*, FSE 2020 | Dependency-based neural program slicing | Temporal source/protocol support is closely related; novelty needs detailed comparison |
| Cortese et al., *Robust, positive and exact model reduction via monotone matrices*, 2025 | Positive exact reduction | Coordinate masks with fixed original dynamics differ from unrestricted replacement realizations |
| Zhang et al., *Perturbation-Tolerant Structural Controllability*, arXiv:2105.00968 | Generic and perturbation-sensitive structural properties | Polynomial nonvanishing is established; candidate contribution is the temporal mask statement and finite-radius obstruction |
| Balwani, *MechInterp for Recurrent Computation: Time-Resolved Circuit Discovery in RNNs*, workshop listing 2026 | Time-window ablation and trajectory/Jacobian analysis | Essential baseline; only listing/abstract reviewed, full OpenReview access blocked |

Bibliographic identifiers and links are in `paper/references.bib`. The paper does not claim that all relevant prior work has been found. In particular, the nonnegative path proof is simple and the mixed finite-difference inequality may have close mathematical antecedents. Independent verification is required before novelty claims are strengthened.
