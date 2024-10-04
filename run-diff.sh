for seed in {1..3}; do
    python train.py -s $seed -ps $seed -t 10000000 -a 100 -e 10 -w -wg 100-resource-diff -r 2
    python train.py -s $seed -ps $seed -t 10000000 -a 100 -e 10 -w -wg 100-resource-diff -r 2 -i
done

for seed in {1..3}; do
    python train.py -s $seed -ps $seed -t 10000000 -a 100 -e 10 -w -wg 100-resource-diff -r 2
    python train.py -s $seed -ps $seed -t 10000000 -a 100 -e 10 -w -wg 100-resource-diff -r 2 -i
done

for seed in {1..3}; do
    python train.py -s $seed -ps $seed -t 10000000 -a 100 -e 10 -w -wg 100-resource-diff -r 3
    python train.py -s $seed -ps $seed -t 10000000 -a 100 -e 10 -w -wg 100-resource-diff -r 3 -i
done

for seed in {1..3}; do
    python train.py -s $seed -ps $seed -t 10000000 -a 100 -e 10 -w -wg 100-resource-diff -r 4
    python train.py -s $seed -ps $seed -t 10000000 -a 100 -e 10 -w -wg 100-resource-diff -r 4 -i
done

for seed in {1..3}; do
    python train.py -s $seed -ps $seed -t 10000000 -a 100 -e 10 -w -wg 100-resource-diff -r 5
    python train.py -s $seed -ps $seed -t 10000000 -a 100 -e 10 -w -wg 100-resource-diff -r 5 -i
done