import pdb
from matplotlib.pyplot import axis
from tqdm import tqdm
import torch
import gc
import os
from myutils import get_ap_score
import numpy as np
import logging
import time
from sklearn.metrics import average_precision_score, accuracy_score 

def train_model(model, device, optimizer, scheduler, train_data, val_data, save_dir, model_num, epochs, log_file, eval_metric):
    tr_loss_list, tr_map_list = [], []
    val_loss_list, val_map_list = [], []
    best_train_map = 0.0
    best_val_map = 0.0
    
    for epoch in range(epochs):
        print("-------Epoch {}----------".format(epoch+1))
        log_file.write("Epoch {} >>".format(epoch+1))
        optimizer.step()
        scheduler.step()
        
        running_loss = 0.0
        total_loss = 0.0
        running_ap = 0.0
        
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        m = torch.nn.Sigmoid()
        soft = torch.nn.Softmax(dim=1)
        
        model.train(True)  # Set model to training mode
        
        for i, batch in enumerate(train_data):
            #print(data)
            # target = target.float()
            # pdb.set_trace()
            data = batch[0]
            label = batch[1]
            box = batch[2]
            img_id = batch[3][0]
            observer_id = batch[4][0]
            fixations = batch[5]
            gt_label = label[0, 0, 4:5].squeeze(axis=-1)
            # gt_label = label[:, :, 4:5].squeeze(axis=-1)
            # gt_label = label[:,:,5:].squeeze(axis=-1)
            # gt_label = gt_label.reshape(1,11)
            gt_label = gt_label.long()
            gt_label = gt_label.reshape(1)
            # gt_label = gt_label[:,:11]
            # print(gt_label)
            gt_box = label[:, :, :4]
            data, gt_label, gt_box, box, fixations = data.to(device), gt_label.to(device), gt_box.to(device), box.to(device), fixations.to(device)
            optimizer.zero_grad()
            cls_pred = model(data,gt_box,box,fixations)
            # print(cls_pred)
            loss = criterion(cls_pred, gt_label)
            #print("LOSS: ",loss)
            running_loss += loss.item() # sum up batch loss
            total_loss += loss
            # print("RUNNING LOSS: ",running_loss)
            running_ap += get_ap_score(gt_label.to('cpu').detach().numpy(), soft(cls_pred).to('cpu').detach().numpy()) 
            loss.backward()
            optimizer.step()
            if i%100==99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
                
                #clear variables
                # del data, target, output
                # gc.collect()
                # torch.cuda.empty_cache()
        #pdb.set_trace()
        train_loss = total_loss/len(train_data)
        running_ap_train = running_ap/len(train_data)
        print("Train Loss: ", train_loss)
        print("Train MAP: ", running_ap_train)

        #if running_ap_train >= best_train_map:
            #best_train_map = running_ap_train
            #log_file.write("Saving best weights...\n")
            #torch.save(model.state_dict(), os.path.join(save_dir,"model-{}.pth".format(model_num)))

        tr_loss_list.append(train_loss.to('cpu').detach().numpy().item())
        tr_map_list.append(running_ap_train)
        #-------------------------Validation---------------------------#
        #pdb.set_trace()        
        #correct = 0
        #total = 0
        #model.train(False)
        #weights_file_path =  os.path.join(save_dir, "model-{}.pth".format(model_num))
        #if os.path.isfile(weights_file_path):
            #print("Loading best weights")
            #model.load_state_dict(torch.load(weights_file_path))

        total_val_loss = 0
        running_ap_val = 0
        with torch.no_grad():
            for batch in val_data:
                cls_scores = []
                gt_classes = []
                data = batch[0]
                label = batch[1]
                box = batch[2]
                img_id = batch[3][0]
                observer_id = batch[4][0]
                fixations = batch[5]
                data, label, box, fixations = data.to(device), label.to(device), box.to(device), fixations.to(device)
                gt_box = label[:, :, :4]
                gt_label = label[0,0,4:5].squeeze(axis = -1)
                gt_label = gt_label.long()
                gt_label = gt_label.reshape(1)
                cls_score = model(data, gt_box, box, fixations)
                val_loss = criterion(cls_score, gt_label)
                total_val_loss += val_loss
                #running_ap_val += get_ap_score(torch.Tensor.cpu(gt_label).detach().numpy(), torch.Tensor.cpu(m(cls_score)).detach().numpy())
                running_ap_val += get_ap_score(gt_label.to('cpu').detach().numpy(), soft(cls_score).to('cpu').detach().numpy())
                #print(running_ap_val)
                #cls_score = mx.nd.softmax(cls_score, axis=-1)
                soft = torch.nn.Softmax(dim=1)
                cls_score = soft(cls_score)
                cls_scores.append(cls_score[:, :])
                gt_classes.append(label[:, :, 5:])
            #pdb.set_trace()
            validation_loss = total_val_loss/len(val_data)
            validation_map = running_ap_val/len(val_data)
            print("Total Validation Loss: ", validation_loss)
            print("Validation mean average precision: ", validation_map) 
            #for score, gt_class in zip(cls_scores, gt_classes):
                #eval_metric.update(score, gt_class)
            #pdb.set_trace()
            #map_name, mean_ap = eval_metric.get()
            #val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            val_msg = '\n'.join(["Validation Loss: {} ".format(validation_loss), "Validation MAP: {} ".format(validation_map)])
            log_file.write('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            #current_map = float(mean_ap[-1])
            #print("Current MAP: ", current_map)
            #outputs_file_name = '{:s}_{:04d}_val_outputs.csv'.format("", epoch)
            #eval_metric.save(file_name=outputs_file_name)
            # zero the parameter gradients
            
            # Get metrics here
        
            # Backpropagate the system the determine the gradients
            
            # Update the paramteres of the model

            val_loss_list.append(validation_loss.to('cpu').detach().numpy().item())
            val_map_list.append(validation_map)

            if validation_map >= best_val_map:
                best_val_map = validation_map
                log_file.write("Saving best weights...\n")
                torch.save(model.state_dict(), os.path.join(save_dir,"model-{}.pth".format(model_num)))
    
            
            
            #print("loss = ", running_loss)
            
        #num_samples = float(len(train_data))
        #tr_loss_ = running_loss.item()/num_samples
        #tr_map_ = running_ap/num_samples
        
        #print('train_loss: {:.4f}, train_avg_precision:{:.3f}'.format(
            #tr_loss_, tr_map_))
        
        #log_file.write('train_loss: {:.4f}, train_avg_precision:{:.3f}, '.format(
            #tr_loss_, tr_map_))
        
        # Append the values to global arrays
        #tr_loss.append(tr_loss_), tr_map.append(tr_map_)
                    
    return ([tr_loss_list, tr_map_list], [val_loss_list, val_map_list])