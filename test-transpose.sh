#!/bin/bash
algo=0
for n in 16 32 64 128 256 512 1024 2048 4096; do 
    ./matrix -n $n -a $algo > tmp.txt
    dur=`grep secs tmp.txt | awk '{ print $3 }'`
    ws=`expr $n \* $n \* 4 \* 3 / 1024`
    echo $n $ws $dur
done
