# Viva Q&A

## What is Stream A?

Stream A is the frame-centric anomaly detection stream in ARGUS. It uses VideoMAE clip embeddings and MULDE anomaly scoring.

## Is it supervised or unsupervised?

It is unsupervised / one-class anomaly detection. Training uses only normal videos. Labels are used only for validation and benchmarking.

## What gap did you solve?

We implemented and improved a frame-centric Avenue path for a standalone MULDE-style system. The paper's famous Avenue score is object-centric, so our setup fills a different practical gap.

## Why is 94.3 not your target?

Because `94.3` is from the paper's object-centric Avenue setup. Our current standalone Stream A is frame-centric, so that is not a direct comparison.

## Then how do you compare your work?

First, we compare our own baseline against our improved result on the same setup. Second, we compare our result to other frame-level Avenue methods only as context.

## What was your baseline and final result?

Baseline Avenue result:

- micro AUC: `0.7738`
- macro AUC: `0.7724`

Main reported result:

- micro AUC: `0.8451`
- macro AUC: `0.8514`

## What changes improved the result?

- correct MULDE scoring surface for Avenue
- GMM-based aggregation
- better regularization and learning rate
- normal-only holdout checkpoint selection
- GMM numerical-stability fixes

## Why did you use holdout selection on Avenue?

Because Avenue does not naturally provide the same anomalous validation setup as UBnormal. For a normal-only training setting, holdout selection is the honest practical choice.

## What is the backbone?

`VideoMAE-v2 Base`

## What is the anomaly scorer?

`MULDE`

## What does the demo show?

The demo takes a video, extracts VideoMAE features, scores anomaly over time with MULDE, and shows a frame-level anomaly timeline plus peak anomaly frames.

## Did you change VideoMAE?

No. VideoMAE stayed fixed. The gains came from engineering and tuning the Avenue MULDE path correctly.

## What is the best number you observed?

The benchmark-safe main result is `0.8451` micro AUC.  
A small post-hoc eval-only diagnostic reached `0.8466`, but the main reported result should remain `0.8451`.

## What would be the next step after this?

The next serious step would be either stronger temporal feature engineering on top of frozen VideoMAE embeddings or a true object-centric branch, but that is future work.
