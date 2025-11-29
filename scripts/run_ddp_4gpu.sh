#!/bin/bash
torchrun --nproc_per_node=4 -m parallel.train_ddp
