#!/bin/bash

# 依次运行 evaluate_pgd.py 10 次
for i in {1..10}
do
   echo "Running evaluate_pgdwrn.py iteration $i"
   python3 evaluate_pgdwrn.py
done

echo "Finished running evaluate_pgdwrn.py 10 times."