import torch
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

from datetime import datetime
import torch.optim as optim
import matplotlib.pyplot as plt
from model import IDCM_NN
from train_model import train_model
from load_data import get_loader
from evaluate import fx_calc_map_label
######################################################################
# Start running

if __name__ == '__main__':
    # environmental setting: setting the following parameters based on your experimental environment.
    dataset = 'pascal'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data parameters
    DATA_DIR = 'data/' + dataset + '/'
    alpha = 1e-3
    beta = 1e-1
    MAX_EPOCH = 500
    batch_size = 100
    # batch_size = 512
    lr = 1e-4
    betas = (0.5, 0.999)
    weight_decay = 0

    print('...Data loading is beginning...')

    data_loader, input_data_par = get_loader(DATA_DIR, batch_size)

    print('...Data loading is completed...')

    model_ft = IDCM_NN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'], output_dim=input_data_par['num_class']).to(device)
    params_to_update = list(model_ft.parameters())

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=lr, betas=betas)

    print('...Training is beginning...')
    # Train and evaluate
    model_ft, img_acc_hist, txt_acc_hist, loss_hist = train_model(model_ft, data_loader, optimizer, alpha, beta, MAX_EPOCH)
    print('...Training is completed...')

    print('...Evaluation on testing data...')
    view1_feature, view2_feature, view1_predict, view2_predict = model_ft(torch.tensor(input_data_par['img_test']).to(device), torch.tensor(input_data_par['text_test']).to(device))
    label = torch.argmax(torch.tensor(input_data_par['label_test']), dim=1)
    view1_feature = view1_feature.detach().cpu().numpy()
    view2_feature = view2_feature.detach().cpu().numpy()
    view1_predict = view1_predict.detach().cpu().numpy()
    view2_predict = view2_predict.detach().cpu().numpy()
    img_to_txt = fx_calc_map_label(view1_feature, view2_feature, label)
    print('...Image to Text MAP = {}'.format(img_to_txt))

    txt_to_img = fx_calc_map_label(view2_feature, view1_feature, label)
    print('...Text to Image MAP = {}'.format(txt_to_img))

    print('...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)))
