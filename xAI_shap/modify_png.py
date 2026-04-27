import numpy as np, pickle as pk, matplotlib.pyplot as plt
# work_dir lysispeptica_thesis_plot
# python /home/cclee/RDDL/lysispeptica_thesis_plot/xAI_shap/modify_png.py



### input setting

#train_folder = 'models/m1_791_836_cnn_zs_5544'  #md_id = 1
#train_folder = 'models/m2_798_796_cnn2_zs_5545'  #md_id = 2
train_folder = 'models/m3_763_843_5950_3p1bn_ugml2std'  #md_id = 3
#train_folder = 'models/m4_843_750_5041chatt_ugml2std'  #md_id = 4

tset = 1 #1 or 2
#==========================================





md_id_dict={'models/m1_791_836_cnn_zs_5544':1, 'models/m2_798_796_cnn2_zs_5545':2
            ,'models/m3_763_843_5950_3p1bn_ugml2std':3, 'models/m4_843_750_5041chatt_ugml2std':4}

md_id=md_id_dict[train_folder]

### output
test_csv   = f'{train_folder}/test{tset}.csv'
plot_title = f'Feature Importance\n(Model {md_id} on Test Set {tset})'
plot_sv    = f'{train_folder}/shap_img/md{md_id}_t{tset}_study.png'
shap_path  = f'{train_folder}/shap_array/md{md_id}_t{tset}.pkl'
#==============================================================




with open(shap_path, 'rb') as f:
    shap_values=pk.load(f)
    print(shap_values.shape) 

    # (173, 49, 161, 2) #batch, seq_len, features, positive&negative_values (only model with softmax output)

pos_class = 0 # 0 , 1
sv_pos = shap_values[..., pos_class] # (184, 49, 8)

# <step 1> Absolute value 
sv_abs = np.abs(sv_pos)

# <step 2> Average across sequence dim (axis=1) -> (184, 8)
sv_seq_avg = sv_abs.mean(axis=1)

# <step 2> sum across sequence dim (axis=1) -> (184, 8)
#sv_seq_avg = sv_abs.sum(axis=1)


# <step 3> mean across samples (global average importance) -> vector of features (8)
feat_importance = sv_seq_avg.mean(axis=0)   # shape (8,)

# <step 3> sum across samples (total attribution mass) -> vector of features (8)
#feat_importance = sv_seq_avg.sum(axis=0)   # shape (8,)

#print(feat_importance.shape)  





def plot_shap_barchart_cnn(feat_importance, plot_sv, plot_title):
    print('\nplotting\n')

    print(feat_importance)
    #feature_names = [f"f{i+1}" for i in range(8)]
    feature_names = ['H1','V','P1','pl','pKa','NCI','C(µg/mL)', 'C(µM)']
    print(feature_names)
    sorted_idx = np.argsort(feat_importance)[::-1]
    print(sorted_idx) 
    # [6 2 0 7 3 1 4 5]
    # [2 6 0 3 1 4 7 5]

    sorted_values = feat_importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
  
    top10_name = sorted_names[:10]
    top10_sv = sorted_values[:10]

    #plt.xlim(0, 0.018)
    #plt.figure(figsize=(10, 20))  # Adjust height for visibility
    plt.barh( top10_name, top10_sv , color='wheat')

    plt.xlabel("Mean|SHAP value|")  
 
    plt.title(f"{plot_title}", fontsize=12)
    #plt.ylabel("Feature", fontsize=8)
    plt.gca().invert_yaxis()  # Highest on top
    plt.savefig(plot_sv, dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close()




def plot_shap_barchart_mlp(feat_importance, plot_sv, plot_title):
    print('\nplotting\n')


    feature_names = [f"f{i+1}" for i in range(160)]
    #feature_names.append('C(µM)')
    feature_names.append('C(µg/ml)')

    sorted_idx = np.argsort(feat_importance)[::-1]
    sorted_values = feat_importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    top10_name = sorted_names[:10]
    top10_sv = sorted_values[:10]

    #plt.xlim(0, 0.018)
    #plt.figure(figsize=(10, 20))  # Adjust height for visibility
    plt.barh( top10_name, top10_sv , color='wheat')

    plt.xlabel("Mean|SHAP value|")  
 
    plt.title(f"{plot_title}", fontsize=12)
    #plt.ylabel("Feature", fontsize=8)
    plt.gca().invert_yaxis()  # Highest on top
    plt.savefig(plot_sv, dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close()




if (md_id == 1) or  ( md_id == 2 ):
    print('run shap_barchart_cnn')
    plot_shap_barchart_cnn(feat_importance, plot_sv, plot_title)
elif (md_id == 3) or  ( md_id == 4 ):
    print('run shap_barchart_mlp')
    plot_shap_barchart_mlp(feat_importance, plot_sv, plot_title)
else:
    print('==error!== invalid md_id number')


'''
[[-0.0070975   0.0070975 ]
 [-0.00211995  0.00211995]
 [-0.00274838  0.00274838]
 [-0.00293785  0.00293785]
 [ 0.02121016 -0.02121016]
 [-0.0034591   0.0034591 ]
 [ 0.00268953 -0.00268953]
 [-0.0029701   0.0029701 ]]
'''



