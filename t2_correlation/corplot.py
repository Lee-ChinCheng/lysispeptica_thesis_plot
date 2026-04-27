import numpy as np, matplotlib.pyplot as plt
from scipy import stats

### goal: plot correlation line, 
# test set 2 predicted prob% (0~1) vs actual hemolysis%(0~50)
#============================================================

### setting
t2_ens_prob = 't2_correlation/prob/t2_ens4md.txt'
t2_hemo     = 'test_set/test_set2.fa' #confidential
img_output  = 't2_correlation/img_output/t2_corr_504.png'
#=======================================================



id_value={}
#get test set 2 predicted prob%
with open(t2_ens_prob, 'r') as f:
    for l in f:
        l=l.strip()
        if l[0]=='(': continue
        sid=l.rsplit('_',2)[0]
        prob=float(l.rsplit('_',1)[-1])
        #GAN-pep1_1_0_0.21275
        id_value[sid]=[prob]

#get test set 2 actual hemolysis%
with open(t2_hemo, 'r') as f:
    for l in f:
        l=l.strip()
        #>GAN-pep1_64	27.19519737_124.2438_64
        if l[0]=='>':
            sid=l.split('\t')[0][1:]
            hemo=float(l.split('\t')[1].split('_')[0])
            if sid in id_value:
                id_value[sid].append(hemo)
            else:
                print('--error--')
                break

#print(len(id_value)) #126
#print(id_value)


### plot
p_li, h_li =[], []
for k,v in id_value.items():
    p_li.append(v[0])
    if v[1] < 0: hemo=0
    else: hemo = v[1]
    h_li.append(hemo)


h_li = np.array(h_li)
p_li = np.array(p_li)

# Correlation coefficient (Pearson)
r, p_value = stats.pearsonr(h_li, p_li) # x is hemo, y is prob

#  Linear regression (best fit line)
slope, intercept = np.polyfit(h_li, p_li, 1)
regression_line = slope * h_li + intercept

plt.figure(figsize=(6, 6))
plt.scatter(h_li, p_li, color='blue', s=5, alpha=0.6)
plt.plot(h_li, regression_line, color='firebrick', linewidth=0.65, label=f"Regression line (r = {r:.3f})")
plt.xlim(-1, 80)
plt.ylim(-0.01, 1.01)
plt.yticks(np.arange(0,  1.01, 0.1))

plt.axhline(0.5, color='dimgray', linewidth=0.5, linestyle='-', alpha = 0.7)
plt.axvline(10, color='dimgray', linewidth=0.5, linestyle='-', alpha = 0.7)


plt.title('Test Set 2\nPredicted Probability vs Actual Hemolysis') #f"Correlation: r = {r:.3f}"
plt.xlabel("Hemolysis%")
plt.ylabel("Probability")
plt.legend()
#plt.grid(True, linestyle='--', alpha=0.35)
plt.savefig(img_output, dpi = 900,  bbox_inches='tight', pad_inches=0.05)
plt.close()

print(f"Correlation coefficient (Pearson r): {r:.4f}")
print(f"P-value: {p_value:.4e}")

