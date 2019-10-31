from diswiz.main_context_predictor import diswiz_model_functions

train_data = '../data/swda-actags_train_speaker.csv'
test_data = '../data/swda-actags_test_speaker.csv'

param_file_nc = '../diswiz/params/params_non_context'
param_file_nc_new = '../diswiz/params/params_non_context_new'
param_file_con = '../diswiz/params/params_context'
param_file_con_new = '../diswiz/params/params_context_new'

diswiz_model_functions(train_data, test_data,
                       param_file_nc, param_file_con,
                       param_file_nc_new, param_file_con_new, train=True)
