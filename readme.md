
### Steps to reproduce the results
1. Run ```main_swda_elmo_mean.py``` for training (set ```train = True```)
 the models.

2. To annotate IEMOCAP data:
run ```mocap_dia_act_annotate.py``` to get context and non-context 
annotations and save as numpy arrays, 
and generate final annotation (csv) file.

3. To annotate MELD data:
run ```meld_dia_act_annotate.py``` to get context and non-context
annotations and save as numpy arrays,
and generate final annotation (csv) file.

