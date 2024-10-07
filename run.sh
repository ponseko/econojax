for pop in {42..44}; do
    for seed in {42..47}; do
        python train.py -s $seed -ps $pop -t 10000000 -a 100 -e 10 -w -g -wg 100-f-pop$pop;
        python train.py -s $seed -ps $pop -t 10000000 -a 100 -e 10 -w -wg 100-f-pop$pop;
    done
done