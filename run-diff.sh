for num_resources in {2..5}; do
    for seed in {1..3}; do
        python train.py -s $seed -ps $seed -t 5000000 -a 50 -e 10 -w -wg resource-$num_resources -r $num_resources --trade_prices 3 6 9
        python train.py -s $seed -ps $seed -t 5000000 -a 50 -e 10 -w -wg resource-$num_resources -r $num_resources -i --trade_prices 3 6 9
    done
done
