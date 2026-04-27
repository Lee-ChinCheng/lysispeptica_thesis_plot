import matplotlib.pyplot as plt
import numpy as np



input_dataset ='./peptide_2984.fa'
output_ugml = 'scatter/img_output/scatter_ugml.png'
output_uM   = 'scatter/img_output/scatter_uM.png'


def collect_hemo_ugml_uM(inp):
    hemo_li, ugml_li, um_li = [],[],[]
    pam=nam=0
    with open(inp, 'r') as f:
            #>13041	10.0_200.548_128.0
            #AARIILRDRFR
            for l in f:
                l=l.strip()
                if l[0]=='>':
                    
                    hemo = float(l.split('\t')[1].split('_')[0])
                    ugml = float(l.split('\t')[1].split('_')[1])
                    uM   = float(l.split('\t')[1].split('_')[2])
                    if hemo > 100:
                        hemo = 100
                        
                    
                    hemo_li.append(hemo)
                    ugml_li.append(ugml)
                    um_li.append(uM)

                    if hemo > 10: pam+=1
                    else: nam +=1
                                  
                else:
                    #seq=l
                    pass
                
    print(len(hemo_li), len(ugml_li)) 
    print(pam,nam) #1391 1593
    return(hemo_li, ugml_li, um_li)



result=collect_hemo_ugml_uM(input_dataset)

#ugml
print( round(np.mean(result[1]), 3),  round(np.std(result[1]), 3), min(result[1]), max(result[1])) #120.448 86.832 0.2 300.0
print( np.percentile(result[1], 95), np.percentile(result[1], 90), np.percentile(result[1], 85) ) #262.56 250.0 235.37
#uM
print( round(np.mean(result[2]), 3),  round(np.std(result[2]), 3), min(result[2]), max(result[2])) #52.601 41.371 0.08 213.402
print( np.percentile(result[2], 95), np.percentile(result[2], 90), np.percentile(result[2], 85) ) #128.0 112.5 100.0



### Plot 1
plt.figure(figsize=(8, 6))
plt.scatter(result[0], result[1], color='blue', s=15, alpha=0.6)
plt.title('Data set scatter plot 1', fontsize=11.5)
plt.xlabel('Hemolysis%', fontsize=11)
plt.ylabel(r'Concentration ( $\mathbf{\mu g/ml}$ )', fontsize=11, color='firebrick')
plt.xlim(-5, 105)
plt.xticks(np.arange(0, 101, 10))   # 0,10,20,...,100
plt.grid(True, linestyle='--', alpha=0.6)
#plt.tight_layout()
plt.savefig(output_ugml, dpi = 600,  bbox_inches='tight', pad_inches=0.05)
plt.close()


### Plot 2
plt.figure(figsize=(8, 6))
plt.scatter(result[0], result[2], color='blue', s=15, alpha=0.6)
plt.title('Data set scatter plot 2 ', fontsize=11.5)
plt.xlabel('Hemolysis%', fontsize=11)
plt.ylabel(r'Concentration ( $\mathbf{\mu M}$ )', fontsize=11, color='firebrick')
plt.ylim(-5, 220)
plt.xlim(-5, 105)
plt.xticks(np.arange(0, 101, 10))   
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(output_uM, dpi = 600,  bbox_inches='tight', pad_inches=0.05)
plt.close()
