#!/bin/bash
# Max sequence identity of each Test Set 1 sequence to the training set.
# train.fa = peptide_1645.fa, test.fa = test_set1_173.fa (headers reduced to sample_id).
# cd-hit-2d writes to nov_<c> the query sequences that have NO match at >= c,
# so a sequence's max-identity band is the highest c at which it disappears.
cd "$(dirname "$0")"
for ct in 0.9:5 0.8:5 0.7:5 0.6:4 0.5:3; do
  c=${ct%%:*}; n=${ct##*:}
  cd-hit-2d -i train.fa -i2 test.fa -o nov_$c -c $c -n $n -l 5 -d 0 -M 0 -T 0 > log_$c.txt 2>&1
  echo "c=$c  no training neighbour: $(grep -c '^>' nov_$c)"
done
