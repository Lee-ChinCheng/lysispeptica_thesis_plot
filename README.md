# lysispeptica_thesis_plot
Generating images for the LysisPeptica thesis, helping visualize concepts and results efficiently.

---
<p align="center">
  <img src="./others/main_concept2.png " alt="main_concept2" width="400" height="508"/>
</p>

* main flowchart of LysisPeptica ensemble model 
* img path: /others/main_concept2.png 
* was plotted by ppt

---
<p align="center">
  <img src="./test_set/ROC_plot/testsets_roc.png" alt="testsets_roc" width="450" height="438"/>
</p>

* overlapped ROC of two test sets 
* img path: /test_set/ROC_plot/testsets_roc.png 
* was plotted by /test_set/ens_plot.py

---
<p align="center">
  <img src="./t2_property/img_output/t2_ens.png" alt="t2_ens" width="550" height="454"/>
</p>

* test set 2 linegraph across different concentration level (1,2,4,8,16,32,64 繕M)
* img path: /t2_property/img_output/t2_ens.png
* was plotted by /t2_property/t2_linegraph.py

---
<p align="center">
  <img src="./models/m1_791_836_cnn_zs_5544/shap_img/md1_t1.png" alt="md1_t1" width="450" height="358"/>
</p>

* SHAP value (feature importance analysis) of model 1 in test set 1
* img path: /models/m1_791_836_cnn_zs_5544/shap_img/md1_t1.png
* was plotted by /xAI_shap/modify_png.py

---
<p align="center">
  <img src="./models/m3_763_843_5950_3p1bn_ugml2std/shap_img/md3_t1.png" alt="md3_t1" width="450" height="360"/>
</p>

* SHAP value (feature importance analysis) of model 3 in test set 1
* img path: /models/m3_763_843_5950_3p1bn_ugml2std/shap_img/md3_t1.png
* was plotted by /xAI_shap/modify_png.py

---
### Other SHAP plots

* Image Path:<br>

|        | test set 1 | test set 2 |
|--------|------------|------------|
| <sub>model1</sub> | <sub>/models/m1_791_836_cnn_zs_5544/shap_img/md1_t1.png</sub> | <sub>/models/m1_791_836_cnn_zs_5544/shap_img/md1_t2.png</sub> |
| <sub>model2</sub> | <sub>/models/m2_798_796_cnn2_zs_5545/shap_img/md2_t1.png</sub>    | <sub>/models/m2_798_796_cnn2_zs_5545/shap_img/md2_t2.png</sub>    |
| <sub>model3</sub> | <sub>/models/m3_763_843_5950_3p1bn_ugml2std/shap_img/md3_t1.png</sub> | <sub>/models/m3_763_843_5950_3p1bn_ugml2std/shap_img/md3_t2.png</sub> |
| <sub>model4</sub> | <sub>/models/m4_843_750_5041chatt_ugml2std/shap_img/md4_t1.png</sub>  | <sub>/models/m4_843_750_5041chatt_ugml2std/shap_img/md4_t2.png</sub>  |
| <sub>ensemble</sub> | <sub>no ensemble model's SHAP plot</sub> | <sub>no ensemble model's SHAP plot</sub> |


* All 8 images were plotted by /xAI_shap/modify_png.py

---
### Model Architectures

<p align="center">
  <img src="./models/m1_791_836_cnn_zs_5544/structure_img/M1_structure.PNG" alt="M1_structure" width="350" height="390"/>
</p>

* Model 1
* img path: /models/m1_791_836_cnn_zs_5544/structure_img/M1_structure.PNG
* was plotted by ppt

<p align="center">
  <img src="./models/m2_798_796_cnn2_zs_5545/structure_img/M2_structure.PNG" alt="M2_structure" width="350" height="390"/>
</p>

* Model 2
* img path: /models/m2_798_796_cnn2_zs_5545/structure_img/M2_structure.PNG
* was plotted by ppt

<p align="center">
  <img src="./models/m3_763_843_5950_3p1bn_ugml2std/structure_img/M3_structure.PNG" alt="M3_structure" width="350" height="555"/>
</p>

* Model 3
* img path: /models/m3_763_843_5950_3p1bn_ugml2std/structure_img/M3_structure.PNG
* was plotted by ppt

<p align="center">
  <img src="./models/m4_843_750_5041chatt_ugml2std/structure_img/M4_structure.PNG" alt="M4_structure" width="410" height="395"/>
</p>

* Model 4
* img path: /models/m4_843_750_5041chatt_ugml2std/structure_img/M4_structure.PNG
* was plotted by ppt


---
<p align="center">
  <img src="./scatter/img_output/scatter_ugml.png" alt="scatter_ugml" width="480" height="374"/>
</p>
<p align="center">
  <img src="./scatter/img_output/scatter_uM.png" alt="scatter_ugml" width="480" height="374"/>
</p>

* Dataset distribution
* img path: /scatter/img_output/scatter_ugml.png
* was plotted by /scatter/data_scatter.py 
* note: /scatter/img_output/scatter_tif.tif was merged by scatter_ugml.png & scatter_uM.png 

---
<p align="center">
  <img src="./t2_correlation/img_output/t2_corr_504.png" alt="t2_corr_504" width="450" height="466"/>
</p>

* test set 2 predicted score(0~1) vs actual hemolysis% <br>
(internal study, not in the thesis)
* img path: /t2_correlation/img_output/t2_corr_504.png
* was plotted by /t2_correlation/corplot.py

---

### Test set fasta file

* test set 1: /test_set/test_set1_173.fa

* test set 2 (sheep): /test_set/test_set2.fa saved in ELN (node/7928#comment-10895) instead of public Github

---

### Peptide property of Test Set
for thesis writing, this file saved Mass, Net charge, Hydrophobicity, Isoelectric point (pI) of peptide

* test set 1: not necessary

* test set 2 (sheep): /t2_property/t2_property.txt saved in ELN (node/7928#comment-10895) instead of public Github


---



