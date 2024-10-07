for seed in {1..3} 
do for num_resources in {2..12..2} 
do
    python train.py -s $seed -ps $seed -t 10000000 -a 40 -e 6 -w -wg $num_resources -r $num_resources --trade_prices 3 6 9 --craft_diff_resources_required 0;
    python train.py -s $seed -ps $seed -t 10000000 -a 40 -e 6 -w -wg $num_resources -r $num_resources -i --trade_prices 3 6 9 --craft_diff_resources_required 0;
    python train.py -s $seed -ps $seed -t 10000000 -a 40 -e 6 -w -wg $num_resources -r $num_resources --insert_agent_ids True --trade_prices 3 6 9 --craft_diff_resources_required 0;
done
done