# Project in TDT05 - Self-Supervised Learning

This repository contains our project for the course TDT05 - Self-Supervised
Learning at NTNU. The project was conducted during the fall of 2023 by Bendik
Holter, Jarl Sondre Sæther, and Jonas Onshuus.

## Contents

This project utilizes SimCLR, a renowned technique within self-supervised
machine learning that creates meaningful representations without using any
labels. More information about the technique can be found in the [original
paper](https://arxiv.org/pdf/2002.05709.pdf).

We used `train_contrastive.py` to train the contrastive model,
`baseline_cnn.ipynb` and `baseline_linear.ipynb` for training the two baselines,
and `downstream.ipynb` for training the downstream model. The training was done on 
IDUN, NTNU's high-performance cluster. The remaining code has
been organized into separate files as they are not as crucial to understanding
the project. An exception to this could be `loss.py`, as it contains some of
the logic for creating negative pairs. The code found in `loss.py` and
`optimizer.py` was inspired by [Spijkervet/SimCLR](https://github.com/Spijkervet/SimCLR).
