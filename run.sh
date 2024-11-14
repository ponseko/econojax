for pop in {1..3}; do
    for seed in {1..5}; do
        python train.py -s $seed -ps $pop -t 10000000 -a 100 -e 10 -w -g -wg 100-final-$pop --init_learned_skills --eval_runs 10;
        python train.py -s $seed -ps $pop -t 10000000 -a 100 -e 10 -w -wg 100-final-$pop --init_learned_skills --eval_runs 10;
    done
done
