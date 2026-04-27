import sys
import numpy as np
from func import show_table, read_label_score_txt, collect_test_id, metric_scores, roc_overlap

### setting

model_list=('models/m1_791_836_cnn_zs_5544', 
            'models/m2_798_796_cnn2_zs_5545',
            'models/m3_763_843_5950_3p1bn_ugml2std',   
            'models/m4_843_750_5041chatt_ugml2std')



#for collect id based on order
t1_csv ='models/m4_843_750_5041chatt_ugml2std/test1.csv' 
t2_csv ='models/m4_843_750_5041chatt_ugml2std/test2.csv' 
output_image = 'test_set/ROC_plot/testsets_roc.png'
#=============================================================


def ens_models_test_by_t1t2(model_list, t1_csv, t2_csv, t1_ens_txt, t2_ens_txt,  save_txt, img_p):  #output prob txt

    pf_li, md_name_li =[],[]
    all_t1_prob, all_t2_prob = [], []
    ens_t1_la, ens_t2_la = [], []
    for i,folder in enumerate(model_list):
        print(i+1, folder)
        md_name_li.append(f't{i+1}')
        md_t1 = f'{folder}/t1_md1.txt'
        md_t2 = f'{folder}/t2_md1.txt'

        md_t1_la, md_t1_prob = read_label_score_txt(md_t1)
        md_t2_la, md_t2_prob = read_label_score_txt(md_t2)

        all_t1_prob.append(md_t1_prob)
        all_t2_prob.append(md_t2_prob)
        ens_t1_la.append(md_t1_la)
        ens_t2_la.append(md_t2_la)

    #print(ens_t1_la[0])
    #print(ens_t1_la[1])
    if np.all(np.array(ens_t1_la) == ens_t1_la[0]): #returns a boolean value (True or False).
        pass
    else:
        print('error--! t1 label lists are not identical')

    if np.all(np.array(ens_t2_la) == ens_t2_la[0]):
        #print('t2 label lists are identical')
        pass
    else:
        print('error--! t2 label lists are not identical')
    
    all_t1_prob = np.array(all_t1_prob)
    ens_t1_prob = all_t1_prob.mean(axis=0)
    mat_d = metric_scores(ens_t1_la[0], ens_t1_prob)
    pf_li.append(mat_d)

    all_t2_prob = np.array(all_t2_prob)
    ens_t2_prob = all_t2_prob.mean(axis=0)
    mat_d = metric_scores(ens_t2_la[0], ens_t2_prob)
    pf_li.append(mat_d)

    # plot ROC
    test_li=('Test Set 1', 'Test Set 2')
    plot_li=[(ens_t1_la[0], ens_t1_prob), (ens_t2_la[0], ens_t2_prob)]
    roc_overlap(test_li, plot_li, img_p)
    # =================================


    print(model_list)
    combined_labels=md_name_li
    str_table = show_table([_.values() for _ in pf_li],
                    headers=pf_li[0].keys(),
                    v_headers=combined_labels,
                    title='', float_fmt='%.3f')
    #=================================================
    #save ens prob txt with id, for further comparison
    #first, collect id by test set order

    if save_txt ==1:
        t1_id = collect_test_id(t1_csv, 't1_csv')
        t2_id = collect_test_id(t2_csv, 't2_csv')
        #test 1
        if len(ens_t1_prob)==len(t1_id):
            with open( t1_ens_txt, 'w' ) as op:
                op.write(str(model_list)+'\n')
                for idx,v in enumerate(ens_t1_la[0]):
                    op.write( t1_id[idx] +'_'+ str(v) +'_'+ str(ens_t1_prob[idx]) +'\n' ) 
        else: sys.exit(" len(ens_t1_prob) != len(t1_id) ")              
        #test 2
        if len(ens_t2_prob)==len(t2_id):
            with open( t2_ens_txt, 'w' ) as op:
                op.write(str(model_list)+'\n')
                for idx,v in enumerate(ens_t2_la[0]):
                    op.write( t2_id[idx] +'_'+ str(v) +'_'+ str(ens_t2_prob[idx]) +'\n' )
        else: sys.exit(" len(ens_t2_prob) != len(t2_id) ")
                      

t1_ens_txt ='Hemo_SnC/ensemble/prob/t1_ens4md.txt'
t2_ens_txt ='Hemo_SnC/ensemble/prob/t2_ens4md.txt'
ens_models_test_by_t1t2(model_list, t1_csv, t2_csv, t1_ens_txt, t2_ens_txt, 1, output_image)

#=======================

'''


('H10_8DL49/p791_836_cnn_zs_5544', 'H10_8DL49/p798_796_cnn2_zs_5545', 
'p792_765_5851_3p1bn_ugml2std',      'H10_161DL49/p843_750_5041chatt_ugml2std')
+---------------------------------------------------------------------+
|                                                                     |
+----+-------+-------+-------+--------+-------+-------+-------+-------+
|    |   Acc |  Spec |  Prec | Recall |    F1 |   MCC | auROC | auPRC |
+----+-------+-------+-------+--------+-------+-------+-------+-------+
| t1 | 0.769 | 0.758 | 0.744 |  0.780 | 0.762 | 0.538 | 0.848 | 0.836 |
| t2 | 0.738 | 0.758 | 0.429 |  0.667 | 0.522 | 0.369 | 0.811 | 0.527 |
+----+-------+-------+-------+--------+-------+-------+-------+-------+

('models/p791_836_cnn_zs_5544', 'models/p798_796_cnn2_zs_5545', 
'models/p763_843_5950_3p1bn_ugml2std', 'models/p843_750_5041chatt_ugml2std')
+---------------------------------------------------------------------+
|                                                                     |
+----+-------+-------+-------+--------+-------+-------+-------+-------+
|    |   Acc |  Spec |  Prec | Recall |    F1 |   MCC | auROC | auPRC |
+----+-------+-------+-------+--------+-------+-------+-------+-------+
| t1 | 0.746 | 0.747 | 0.726 |  0.744 | 0.735 | 0.491 | 0.846 | 0.829 |
| t2 | 0.746 | 0.758 | 0.442 |  0.704 | 0.543 | 0.399 | 0.824 | 0.553 |
+----+-------+-------+-------+--------+-------+-------+-------+-------+




+-------------------------------------------------------------------+
| testset 1  2 #model 1                                                       
+--+-------+-------+-------+--------+-------+-------+-------+-------+
|  |   Acc |  Spec |  Prec | Recall |    f1 |   MCC | auROC | auPRC |
+--+-------+-------+-------+--------+-------+-------+-------+-------+
|  | 0.740 | 0.703 | 0.703 |  0.780 | 0.740 | 0.484 | 0.791 | 0.746 |
+--+-------+-------+-------+--------+-------+-------+-------+-------+
|  | 0.722 | 0.717 | 0.417 |  0.741 | 0.533 | 0.387 | 0.836 | 0.616 |
+--+-------+-------+-------+--------+-------+-------+-------+-------+

2026-02-03 14:10:12 
 3pool conca BN dense(256,128,32)
Baseline
batch_size: 2048
dropout_rate: 0.275
epochs: 110
initial_lr: 0.002
decay_rate: 0.94
bottom_lr: 0.0002
random_seed: 20302
train 0.512 val 0.579
+-------------------------------------------------------------------+
| testset 1 & 2      #Model 3                                               
+--+-------+-------+-------+--------+-------+-------+-------+-------+
|  |   Acc |  Spec |  Prec | Recall |    f1 |   MCC | auROC | auPRC |
+--+-------+-------+-------+--------+-------+-------+-------+-------+
|  | 0.740 | 0.813 | 0.761 |  0.659 | 0.706 | 0.479 | 0.792 | 0.761 | #candi 740acc
+--+-------+-------+-------+--------+-------+-------+-------+-------+
|  | 0.810 | 0.919 | 0.579 |  0.407 | 0.478 | 0.374 | 0.765 | 0.547 |
+--+-------+-------+-------+--------+-------+-------+-------+-------+



+-------------------------------------------------------------------+
| testset 1 & 2      #Model 4                                                        
+--+-------+-------+-------+--------+-------+-------+-------+-------+
|  |   Acc |  Spec |  Prec | Recall |    f1 |   MCC | auROC | auPRC |
+--+-------+-------+-------+--------+-------+-------+-------+-------+
|  | 0.786 | 0.769 | 0.759 |  0.805 | 0.781 | 0.573 | 0.843 | 0.832 |  
+--+-------+-------+-------+--------+-------+-------+-------+-------+
|  | 0.714 | 0.758 | 0.385 |  0.556 | 0.455 | 0.278 | 0.750 | 0.442 | # acc > 0.71
+--+-------+-------+-------+--------+-------+-------+-------+-------+




'''










