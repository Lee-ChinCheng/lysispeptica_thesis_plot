import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, auc,
    matthews_corrcoef, roc_curve,
    precision_recall_curve, 
)



# show_table return string
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


def read_label_score_txt(file1):
    lali,probli=[],[]
    with open(file1, 'r') as f:
        for l in f:
            l=l.strip()
            label=int(l.split('_')[0])
            prob=float(l.split('_')[1])
            lali.append(label)
            probli.append(prob)
    lali=np.array(lali) 
    probli=np.array(probli)  
    return (lali, probli)


def collect_test_id(t1_csv, suffix): #return id_list
    #collect id by test set order
    # if suffix=='t1_csv' -> negative id add '-' in the start
    # if suffix=='t2_csv' -> do nothing
    t1_id =[]
    with open(t1_csv, 'r') as f:
        #sample,label,weight
        #Hemo_predi/Data_pk/pc6/8DL49norm_both_/13717.pickle,1,1.0
        row=pam=nam=0
        for l in f:
            row+=1
            if row <2: continue
            l=l.strip()
            sid=l.split('/')[-1].split('.')[0]
            la=l.split('/')[-1].split(',')[1]
            if la=='0': 
                if suffix=='t1_csv': sid='-'+sid    
                nam+=1
            else:
                pam+=1
            t1_id.append(sid)
    print('collect id test1 P N',pam,nam)
    return t1_id


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



def roc_overlap(test_li, plot_li, img_p):
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], ':', color='gainsboro', linewidth=1, alpha=0.9)

    ls_li=('-','-.', '--')

    for idx,tname in enumerate(test_li):
        fpr, tpr, _ = roc_curve(plot_li[idx][0], plot_li[idx][1])
        auroc = round(auc(fpr, tpr),3)
        plt.plot(fpr, tpr, linewidth=2, linestyle=ls_li[idx%3], label=f'{tname} (auROC = {auroc})')

    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.xticks(np.arange(0.0, 1.2, 0.2), fontsize=11)
    plt.yticks(np.arange(0.0, 1.2, 0.2), fontsize=11)
    plt.gca().set_xticklabels(['%d%%' % (_ * 100) for _ in plt.gca().get_xticks()])
    plt.gca().set_yticklabels(['%d%%' % (_ * 100) for _ in plt.gca().get_yticks()])

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Test Set ROC Curve', fontsize=14.5)
    plt.legend(loc="lower right", fontsize = 12)
    plt.savefig(img_p, dpi = 800,  bbox_inches='tight', pad_inches=0.05)
    plt.close()





