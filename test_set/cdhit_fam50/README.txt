Family (similarity-cluster) assignment used for cdhit_cluster_id / cdhit_cluster_size /
cdhit_cluster_train_n in ../test_set1_173_eval.csv.

curated_1818.fa  = the curated dataset, 1,645 training (peptide_1645.fa)
                   + 173 Test Set 1 (test_set1_173.fa). Header = sample_id only
                   (sign retained: leading '-' = negative). Verified to contain
                   exactly the same 1,818 ids as
                   /home/cclee/RDDL/cdhit_op/study/t10L49_{NP,PN}_customize.fa
                   (868 positive + 950 negative).

command:
  cd-hit -i curated_1818.fa -o fam50 -c 0.5 -n 3 -l 5 -d 0 -M 0 -T 0

result: 448 clusters, 194 singletons, largest cluster 95 members.
        (-c 0.9 on the same input gives 1,599 clusters / almost all singletons,
         because >=90% similar sequences were already removed during curation,
         so 0.9 cannot define head/tail.)

cdhit_cluster_train_n = cluster size minus the number of its members that fall in
Test Set 1, i.e. how many training samples the cluster contributes. This is the
direct analogue of MOSAIC's "ordered by training-set size".
