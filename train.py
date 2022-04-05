from trainer import train
import json


path_results = 'pretrained_models'

# Load params with the parameters reported in the article for each dataset
train_params = json.load(open('./pretrained_models/DVS128_10_24ms/all_params.json', 'r'))
# train_params = json.load(open('./pretrained_models/DVS128_11_24ms/all_params.json', 'r'))
# train_params = json.load(open('./pretrained_models/SLAnimals_3s_48ms/all_params.json', 'r'))
# train_params = json.load(open('./pretrained_models/SLAnimals_4s_48ms/all_params.json', 'r'))

train_params['logger_params']['csv']['save_dir'] = '{}'
for k,v in train_params['callbacks_params']:
    if k != 'model_chck': continue
    v['dirpath'] = '{}/weights/'
    v['filename'] = '{epoch}-{val_loss_total:.5f}-{val_loss_clf:.5f}-{val_acc:.5f}'


# train_params['data_params']['batch_size'] = 4

path_model = train('/tests', path_results, 
                               data_params = train_params['data_params'], 
                               backbone_params = train_params['backbone_params'],
                               clf_params = train_params['clf_params'], 
                               training_params = train_params['training_params'], 
                               optim_params = train_params['optim_params'], 
                               callback_params = train_params['callbacks_params'], 
                               logger_params = train_params['logger_params'])

