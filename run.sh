for pop in {1..3}; do
    for seed in {1..5}; do
        python train.py -s $seed -ps $pop -t 10000000 -a 100 -e 10 -w -g -wg shared100-f-pop$pop;
        python train.py -s $seed -ps $pop -t 10000000 -a 100 -e 10 -w -wg shared100-f-pop$pop;
    done
done