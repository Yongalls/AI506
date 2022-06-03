#!/bin/bash
for ((i=0; i<7800; i+=200));
do
  echo from $i to `expr $i + 200`
  python3 completion.py frequent_items --start=$i --end=`expr $i + 200` --filename=$i
done

python3 completion.py frequent_items --start=7800 --filename=7800
