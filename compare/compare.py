import sys
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, auc,
    matthews_corrcoef, roc_curve,
    precision_recall_curve, 
)

### input setting 
t1_toxinpred = 'compare/toxinpred3.0/test1_thr050.csv'
t2_toxinpred = 'compare/toxinpred3.0/test2_thr050.csv'
t2_basic_info='test_set/test_set2.fa'
#=====================================



def metric_scores(y_true, y_pred): #return dic
    ### double check the input y_pred
    #is already    [1,0,0,1]
    #or still prob [0.6, 0.4, 0.3, 0.5]
    y_pred_class = np.around(y_pred)
    #y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    #'auROC_v2': roc_auc_score(va_lali, y_pred_prob),
    #'auPRC_v2': average_precision_score(va_lali, y_pred_prob)
    return {'Acc': round(accuracy_score(y_true, y_pred_class),3),
            'Spec': round(recall_score(y_true, y_pred_class, pos_label=0),3),
            'Prec': round(precision_score(y_true, y_pred_class, zero_division=0),3),
            'Recall': round(recall_score(y_true, y_pred_class),3),
            'F1': round(f1_score(y_true, y_pred_class, zero_division=0),3),
            'MCC': round(matthews_corrcoef(y_true, y_pred_class),3),
            'auROC': round(auc(fpr, tpr),3),
            'auPRC': round(auc(recall, precision),3)         
            }


# show_table & return string
def show_table(values, headers=None, v_headers=None, title=None, float_fmt='%.3f'):
    txt=''
    values = [list(_) for _ in values]

    if headers is not None:
        headers = list(headers)
        item_widths = [len(_) for _ in headers]
    else:
        item_widths = [0 for _ in range(len(values[0]))]

    if v_headers is not None:
        if headers is not None:
            headers.insert(0, '')
        for i, row in enumerate(values):
            row.insert(0, v_headers[i])
        item_widths.insert(0, 0)
    
    for row in values:
        for i, v in enumerate(row):
            row[i] = float_fmt % v if isinstance(v, float) else str(v)
            item_widths[i] = max(item_widths[i], len(row[i]))

    sep_line = '+%s+' % '+'.join('-' * (w + 2) for w in item_widths)

    if title is not None:
        print('+%s+' % ('-' * (len(sep_line) - 2)))
        print('| %%-%ds |' % (len(sep_line) - 4) % title)
        txt = txt + ('+%s+' % ('-' * (len(sep_line) - 2))) +'\n'
        txt = txt + ('| %%-%ds |' % (len(sep_line) - 4) % title) +'\n'
    
    print(sep_line)
    txt = txt + sep_line +'\n'

    if headers is not None:
        #print(headers) #['', 'Acc', 'Spec', 'Prec', 'Recall', 'f1', 'MCC', 'auROC', 'auPRC']
        #print(item_widths) #[0, 5, 5, 5, 6, 5, 5, 5, 5]
        for i, h in enumerate(headers):
            #print(i)
            #print(h)
            print('| %%%ds ' % item_widths[i] % h, end='')
            txt = txt + ('| %%%ds ' % item_widths[i] % h)
                   
        print('|')
        txt = txt + '|\n'
        print(sep_line)
        
        txt = txt + sep_line +'\n'

    for row in values:
        for i, v in enumerate(row):
            print('| %%%ds ' % item_widths[i] % v, end='')
            txt = txt + ('| %%%ds ' % item_widths[i] % v)
        print('|')
        txt = txt + '|\n'
    print(sep_line)
    txt = txt + sep_line +'\n'
    #print(txt)
    return txt


def get_hemolysis_from_t2(t2_basic_info):
    #>GAN-pep1_64	27.2_124.24
    # 64 uM, 27.2 hemolysis%, 124.24 ug/ml
    t2_d={}
    with open(t2_basic_info, 'r') as f:
        for l in f:
            l=l.strip()
            if l[0]=='>':
                sid, hemo = l.split('\t')[0][1:],  round( float(l.split('\t')[1].split('_')[0]) , 3)
                #splitting from the right and dropping the last part.ex: GAN-pep1_64 -> GAN-pep1
                ugml = round( float(l.split('\t')[1].split('_')[1]) , 3)
                prefix = sid.rsplit('_', 1)[0]
                t2_d[ sid ]=(hemo, ugml, prefix)
    print('len(t2_d)', len(t2_d))
    return t2_d

t2_d = get_hemolysis_from_t2(t2_basic_info)
#print(t2_d)

def read_toxinpred_op(hemo_thr, test_num, t2_d, t1_toxinpred):
  
    print('hemolysis thr', hemo_thr)
    ytrue, ypred =[],[]
    r=pam=nam=0
    if test_num==2:
        print('testset2')
        #GAN-pep5_64,ALW,0.38,Non-Toxin,0.350758
        #GAN-pep6_1,FLPI,0.61,Toxin,0.634601
        with open(t1_toxinpred, 'r') as f:
            for l in f:
                l=l.strip().split(',')
                r+=1
                if r<2: continue
                sid, score =l[0], float(l[2])

                #'GAN-pep1_64': (27.195, 124.244, 'GAN-pep1')
                if t2_d[sid][0] > hemo_thr: 
                    ytrue.append(1)
                    pam+=1
                else:                       
                    ytrue.append(0)
                    nam+=1

                ypred.append( score )

    else:
        print('testset1')
        #6702,FLPFLLSALPKVFCFFSKKC,0.73,Toxin,0.782693
        #-2159,ILGKIWEGIKSLF,1.0,Toxin,1.0
        with open(t1_toxinpred, 'r') as f:
            for l in f:
                l=l.strip().split(',')
                r+=1
                if r<2: continue
                sid, score =l[0], float(l[2])

                if sid[0]=='-': ytrue.append(0)
                else:           ytrue.append(1)
  
                ypred.append( score )
    print('true label P N', pam,nam)
    mtx = metric_scores(ytrue, ypred)
    return(mtx)


t1_toxinpred_mtx = read_toxinpred_op(10, 1, t2_d, t1_toxinpred)
t2_toxinpred_mtx = read_toxinpred_op(10, 2, t2_d, t2_toxinpred)

print(t1_toxinpred_mtx)
#{'Acc': 0.543, 'Spec': 0.286, 'Prec': 0.511, 'Recall': 0.829, 'F1': 0.633, 'MCC': 0.136, 'auROC': 0.575, 'auPRC': 0.534}
print(t2_toxinpred_mtx)
#{'Acc': 0.5, 'Spec': 0.465, 'Prec': 0.243, 'Recall': 0.63, 'F1': 0.351, 'MCC': 0.078, 'auROC': 0.512, 'auPRC': 0.206}


