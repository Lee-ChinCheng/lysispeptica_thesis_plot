import numpy as np, pickle as pk, pandas as pd, matplotlib.pyplot as plt
import tensorflow as tf #tensorflow  2.16.1
import os, csv, shap #shap 0.46.0 or 0.48.0
# conda install -c conda-forge   "matplotlib"   "numpy=1.26.4"   --freeze-installed  --dry-run 
# work_dir lysispeptica_thesis_plot
# python /home/cclee/RDDL/lysispeptica_thesis_plot/xAI_shap/run_shap.py



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
model_path = f'{train_folder}/Baseline_model_1.h5'
test_csv = f'{train_folder}/test{tset}.csv'
plot_title=f'Feature Importance\n(Model {md_id} on Test Set {tset})'
plot_sv    = f'{train_folder}/shap_img/md{md_id}_t{tset}.png'


s_img_f, s_array_f  = f'{train_folder}/shap_img', f'{train_folder}/shap_array'
if not os.path.exists(s_img_f):     os.makedirs(s_img_f)
if not os.path.exists( s_array_f):  os.makedirs(s_array_f)
#========================================================




tg_f = 1 # training_1.csv -> be testset, not training set
train_sample_li = []
for filename in os.listdir(train_folder):

    if filename.startswith("training_") and filename.endswith(".csv"):

        if filename == f"training_{tg_f}.csv": continue
        
        file_path = os.path.join(train_folder, filename)
        print(file_path)

        with open(file_path, newline="", encoding="utf-8") as f:
            #sample,label,weight
            #Hemo_predi/Data_pk/pc6/8DL49norm_both_/20122.pickle,1,1.0
            reader = csv.reader(f)
            next(reader)  # Skip header row

            for row in reader:
                if len(row) > 0:  # make sure row is not empty
                    train_sample_li.append(row[0])  

print("training set collected:", len(train_sample_li))





class CustomModel(tf.keras.Model):
        def __init__(self, **kwargs):
            super(CustomModel, self).__init__(**kwargs)

class GlobalMinPooling1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalMinPooling1D, self).__init__(**kwargs)

    def call(self, inputs):
        # reduce along the time/sequence axis
        return tf.reduce_min(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        # input_shape = (batch_size, steps, features)
        return (input_shape[0], input_shape[2])

    def get_config(self):
        config = super(GlobalMinPooling1D, self).get_config()
        return config



def load_model(file_path):

    ### inline Python lambda, and Keras refuses to deserialize it for safety reasons
    #def GlobalMinPooling1D():
    #    return tf.keras.layers.Lambda(lambda x: tf.reduce_min(x, axis=1),   
    #                        output_shape=lambda input_shape: (input_shape[0], input_shape[2]))
    
    return tf.keras.models.load_model(file_path,
                                      custom_objects={'CustomModel': CustomModel, 'tf': tf,
                                                      'GlobalMinPooling1D':GlobalMinPooling1D
                                                     },
                                      compile=False)


def get_array_from_list(global_path_pram:int, train_sample_li: str, num_rows: int = None):
    sample_li=[]
    
    train_sample_li=train_sample_li[:num_rows]
    print(len(train_sample_li))

    for i in train_sample_li:
        #Hemo_predi/Data_pk/pc6/8DL49norm_both_/20122.pickle
        
        if global_path_pram==1:  i='/home/cclee/RDDL/'+i

        with open(i, 'rb') as f:
            arr = pk.load(f)
            sample_li.append(arr)
    single_array = np.stack(sample_li) #be single np arr
    print(single_array.shape)
    print(type(single_array))
    
    return single_array

def get_array( global_path_pram:int, csv_path: str, num_rows: int = None ):

    df = pd.read_csv(csv_path) #pandas skips the header automatically
    # df.iloc[:300, 0] :300 -> first 300 rows ; 0-> first column
    first_col = df.iloc[:, 0]  


    if num_rows is not None:
        pk_path_li = first_col.iloc[:num_rows].tolist()
    else:
        pk_path_li = first_col.tolist()
    
    sample_li=[]
    print('len(pk_path_li)', len(pk_path_li))
    for i in pk_path_li:
        #Hemo_predi/Data_pk/pepb_f/small_uniparc_nm_conc_161D/6369.pickle
        #/home/cclee/RDDL/
        if global_path_pram==1:  i='/home/cclee/RDDL/'+i

        with open(i, 'rb') as f:
            arr = pk.load(f)
            sample_li.append(arr)
    single_array = np.stack(sample_li) #be single np arr
    print(single_array.shape)
    print(type(single_array))
    
    return single_array






background   = get_array_from_list(0, train_sample_li, 665) #(0, train_sample_li, 665) 
test_samples = get_array(0, test_csv)
#-------------
#(800, 49, 8)
#(184, 49, 8)


### run SHAP main program

model = load_model( model_path )
explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(test_samples)
print('shap_values.shape', shap_values.shape) 
# shap_values.shape (184, 49, 8, 2)
with open(f"{train_folder}/shap_array/md{md_id}_t{tset}.pkl", "wb") as f:
    pk.dump(shap_values, f)



pos_class = 0 # 0 , 1
sv_pos = shap_values[..., pos_class] # (184, 49, 8)

# <step 1> Absolute value 
sv_abs = np.abs(sv_pos)

# <step 2> Average across sequence dim (axis=1) -> (184, 8)
#sv_seq_avg = sv_abs.mean(axis=1)

# <step 2> sum across sequence dim (axis=1) -> (184, 8)
sv_seq_avg = sv_abs.sum(axis=1)


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