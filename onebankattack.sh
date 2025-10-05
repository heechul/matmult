#!/bin/bash
# One Bank Attack - Single core matrix multiplication
# Usage: ./onebankattack.sh

# run the onebank write attacker on all other cores
# n_attackers=$(expr $(nproc) - 1)

matvec_victim() 
{
    taskset -c 0 ./matvec -c 32768 -r 8192 -a 0 | grep matvec # algo 0: naive
}
matrix0_victim() 
{
    taskset -c 0 ./matrix -n 2048 -a 0 | grep matmult # algo 0: i-j-k
}

matrix1_victim() 
{
    taskset -c 0 ./matrix -n 2048 -a 1 | grep matmult  # algo 1: i-k-j
}

bkpllread_victim()
{
    pll -f map.txt -e 0 -c 0  -l 16 -g 1 -i 10 -a read -u 128 | grep "MB/s"
}

bandwidth_victim()
{
    # bandwidth benchmark victim
    bandwidth -c 0 -m 65536 -a read -t 1 | grep "MB/s"
}

latency_victim()
{
    # latency benchmark victim
    pll -c 0 -l 1 -m 65536 -u 64 -a read -i 10
}

# victim_func=matrix1_victim
victim_func=bandwidth_victim

# attack types: "write" "read"
for attk in "write"; do
    attacker_pids=()
    for c in $(seq 1 3); do
        pll -f map.txt -e 0 -c $c  -l 8 -g 1 -i 100000 -a $attk -u 64 >& log-attack$c.txt &
        pid=$!
        attacker_pids+=($pid)
        echo "Started attacker on core $c with PID: $pid"
    done
    # for c in $(seq 4 6); do
    #     pll -f map.txt -e 1 -c $c  -l 16 -g 1 -i 100000 -a $attk -u 128 2>&1 &
    #     attacker_pids+=($!)
    #     echo "Started attacker on core $c with PID: $!"
    # done
    n_attackers=$(echo ${attacker_pids[@]} | wc -w)
    echo "All $n_attackers attackers started: PIDs ${attacker_pids[@]}"
    sleep 30 # give the attacker some time to start

    for p in ${attacker_pids[@]}; do
        echo "w/${n_attackers}_attacker(s) doing $attk"
        # call the victim
        $victim_func
        ################################################################################
        # kill one attacker at a time
        sudo kill -9 $p
        n_attackers=$(expr $n_attackers - 1)
        sleep 4 # wait for a while
    done

    echo "Solo run"
    $victim_func
    ################################################################################
done    

