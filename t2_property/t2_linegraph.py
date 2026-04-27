import matplotlib.pyplot as plt, numpy as np
# python /home/cclee/RDDL/lysispeptica_thesis_plot/t2_property/t2_linegraph.py

### setting

model_list=('models/m1_791_836_cnn_zs_5544', 
            'models/m2_798_796_cnn2_zs_5545',
            'models/m3_763_843_5950_3p1bn_ugml2std',   
            'models/m4_843_750_5041chatt_ugml2std')

output_image = 't2_property/img_output/t2_ens.png'
#---------------------------------------


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


def plot_sheep_conc_trend(model_list ):
 
    pep_li=[]
    all_t2_prob, ens_t2_la =[],[]
    for i in range(18):
        pep_li.append( f'AMP_{i+1}' )

    for i,folder in enumerate(model_list):
        print(i+1, folder)
        #md_t1 = f'{folder}/t1_md1.txt'
        md_t2 = f'{folder}/t2_md1.txt'

        md_t2_la, md_t2_prob = read_label_score_txt(md_t2)

        all_t2_prob.append(md_t2_prob)
        #ens_t2_la.append(md_t2_la)

    all_t2_prob = np.array(all_t2_prob)
    ens_t2_prob = all_t2_prob.mean(axis=0)
    

    drug_value_d={}
    for idx,pep in enumerate(pep_li): 
        #print( pep, 7*idx, 7*(idx+1) )
        drug_value_d[pep] = ens_t2_prob[ 7*idx :7*(idx+1)]
        
    x_values = [1,2,4,8,16,32,64]

    nm=0
    for drug_name, ens_t2_prob in drug_value_d.items():
        nm+=1
        if nm%3==0:
            plt.plot(x_values, ens_t2_prob, marker='.', linestyle='--',  label=drug_name)
        elif nm%3==1:
            plt.plot(x_values, ens_t2_prob, marker='*', linestyle=':',  label=drug_name)
        else:
            plt.plot(x_values, ens_t2_prob, marker='x', linestyle=':', label=drug_name)


    plt.xlim(0, 85)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.axhline(0.5, color='dimgray', linewidth=0.7, linestyle='-', alpha = 0.65)
    plt.xlabel("Concentration (µM)", fontsize=10.5)
    plt.ylabel("Probability", fontsize=10.5)
    plt.title("Test Set 2 Prediction\n(our wet-lab validated data)", fontsize=11)
    plt.legend(fontsize=8.5)
    plt.savefig(output_image, dpi=900, bbox_inches='tight', pad_inches=0.05)
    plt.close()
   


plot_sheep_conc_trend(model_list )


