
### Steps to reproduce the results
#### Training and usege of the variant models 
Run ```main_swda_elmo_mean.py``` for training (set ```train = True```)
 the models, else it will return the context and non-context models 
 with direct function ```predict_classes_elmo_mean_features```.
 
 Similarly, ```main_predictor_online.py``` is for models of 
 ELMo normal features.

They are already called in the following scripts for their _usage_, 
and hence before running following scripts, 
directory ```params``` should contain all the weight files and 
directory ```features``` should contain all the elmo features.

#### Annotating the emotion corpora with dialogue acts
1. To annotate IEMOCAP data:
run ```mocap_dia_act_annotate.py``` to get context and non-context 
annotations and save as numpy arrays, 
and generate final annotation (csv) file.

2. To annotate MELD data:
run ```meld_dia_act_annotate.py``` to get context and non-context
annotations and save as numpy arrays,
and generate final annotation (csv) file.

