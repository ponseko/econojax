for num_resources in {2..16..2} 
do for seed in {1..3} 
do
    python train.py -s $seed -ps $seed -t 10000000 -a 25 -e 4 -w -wg run-nr-$num_resources -r $num_resources --trade_prices 3 6 9 --craft_diff_resources_required $(( (num_resources / 2)));
    python train.py -s $seed -ps $seed -t 10000000 -a 25 -e 4 -w -wg run-nr-$num_resources -r $num_resources -i --trade_prices 3 6 9 --craft_diff_resources_required $(( (num_resources / 2)));
    python train.py -s $seed -ps $seed -t 10000000 -a 25 -e 4 -w -wg run-nr-$num_resources -r $num_resources --insert_agent_ids True --trade_prices 3 6 9 --craft_diff_resources_required $(( (num_resources / 2)));
done
done
