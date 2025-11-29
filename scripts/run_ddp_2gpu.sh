#!/bin/bash
torchrun --nproc_per_node=2 -m parallel.train_ddp
