#!/bin/bash
#BSUB -J TrainDemo
#BSUB -q HPC.S1.GPU.X795.sha
#BSUB -n 32
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -gpu num=4

python train-batch32.py
