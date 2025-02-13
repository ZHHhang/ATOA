import torch
import torch.nn as nn
import CNN_2d as AE
import os
import argparse
from data_loader import load_dataset_mask

import matplotlib.pyplot as plt

def main(args):	
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)


    encoder=AE.Encoder_CNN_2D()
    decoder=AE.Decoder_CNN_2D()
    if args.preload_encoder_decoder:
        encoder.load_state_dict(torch.load('./models/test/no_Relu/cae_encoder_cnn_2d_epoch_400_average loss_0.000636_Validation average loss_0.001110.pkl'))
        print("load_the_pre-trained_encoder_model")
        decoder.load_state_dict(torch.load('./models/test/no_Relu/cae_decoder_cnn_2d_epoch_400_average loss_0.000636_Validation average loss_0.001110.pkl'))
        print("load_the_pre-trained_decoder_model")
        # assert 0
	

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
    else:
        assert 0

    enviro_mask_set=load_dataset_mask()



    # criterion = nn.BCEWithLogitsLoss()
    # criterion_test = nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss()
    criterion_test = nn.MSELoss()
    params = list(encoder.parameters())+list(decoder.parameters())
   
    optimizer = torch.optim.Adam(params,lr=args.learning_rate) 
   
    encoder.train()
    decoder.train()
    for epoch in range(args.num_epochs):
        print ("epoch" + str(epoch))
        avg_loss=0
        # for i in range(0, len(enviro_mask_set)-5000, args.batch_size): 
        for i in range(0, len(enviro_mask_set), args.batch_size): 

            if i+args.batch_size<len(enviro_mask_set):
                inp = enviro_mask_set[i:i+args.batch_size]
            else:
                inp = enviro_mask_set[i:]
            inp = torch.from_numpy(inp).float()
            inp=inp.cuda()
            
            temp=encoder(inp.unsqueeze(1))
            outputs = decoder(temp)
            
            loss = criterion(outputs, inp.unsqueeze(1))  
            avg_loss += loss.item()
        
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()  

        print ("--average loss:")
        print (avg_loss/((len(enviro_mask_set)-5000)/args.batch_size))
        if epoch%args.test_save_epoch==0:
            avg_loss_test=0
            # for i in range(len(enviro_mask_set)-5000, len(enviro_mask_set), args.batch_size):
            #     if i+args.batch_size<len(enviro_mask_set):
            #         inp_test = enviro_mask_set[i:i+args.batch_size]
            #     else:
            #         inp_test = enviro_mask_set[i:]
            #     inp_test = torch.from_numpy(inp_test).float()
            #     inp_test = inp_test.cuda()
            #     # ===================forward=====================
            #     temp_test=encoder(inp_test.unsqueeze(1))
            #     outputs_test = decoder(temp_test)
            #     loss_test = criterion_test(outputs_test, inp_test.unsqueeze(1))  # 计算损失
            #     avg_loss_test += loss_test.item()  #change by zheng 

            # print ("--Validation average loss:")
            # print (avg_loss_test/(5000/args.batch_size)) 

            model_path_en='cae_encoder_cnn_2d_epoch_%d_average loss_%f_Validation average loss_%f.pkl'%(epoch,avg_loss/((len(enviro_mask_set)-5000)/args.batch_size),avg_loss_test/(5000/args.batch_size))
            model_path_de='cae_decoder_cnn_2d_epoch_%d_average loss_%f_Validation average loss_%f.pkl'%(epoch,avg_loss/((len(enviro_mask_set)-5000)/args.batch_size),avg_loss_test/(5000/args.batch_size))
            torch.save(encoder.state_dict(),os.path.join(args.model_path,model_path_en))
            torch.save(decoder.state_dict(),os.path.join(args.model_path,model_path_de))
      

         

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/test',help='path for saving trained models')
    parser.add_argument('--num_epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--test_save_epoch', type=int, default=100)
    parser.add_argument('--preload_encoder_decoder', type=int, default=1)
    args = parser.parse_args()
    print(args)
    main(args)


