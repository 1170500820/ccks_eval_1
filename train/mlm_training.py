# https://huggingface.co/transformers/model_doc/bert.html#bertformaskedlm
import sys
sys.path.append('..')

from transformers import BertTokenizer, BertForMaskedLM, AdamW
import torch
import fire
from settings import *
import pickle
import time


def train_mlm(lr=3e-5, epoch=20, save_epoch_cnt=2, save_batch_cnt=500, mult_batch=True, mult_cnt=batch_mult_cnt):
    model = BertForMaskedLM.from_pretrained('../' + model_path)
    masked_ids, token_type_idss, attention_masks, input_idss = \
        pickle.load(open('../preprocess/train_data_for_mlm.pk', 'rb'))
    batch_cnt = len(masked_ids)
    print('data preparation finished, ' + str(len(masked_ids)) + ' batch in total')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    for i_epoch in range(epoch):
        model.train()
        epoch_total_loss = 0.0
        temp_batch_cnt = 0
        accu_loss = None
        epoch_run_time = 0
        for i_batch, masked_batch in enumerate(masked_ids):
            time_start = time.time()
            masked_batch = masked_batch.cuda()
            input_ids = input_idss[i_batch].cuda()
            token_type_ids = token_type_idss[i_batch].cuda()
            attention_mask = attention_masks[i_batch].cuda()
            if mult_batch:
                outputs = model(input_ids=masked_batch, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                labels=input_ids)
                if accu_loss is None:
                    accu_loss = outputs[0]
                else:
                    accu_loss = accu_loss + outputs[0]
                temp_batch_cnt += 1
                if temp_batch_cnt >= mult_cnt:
                    temp_batch_cnt = 0
                    accu_loss.backward()
                    optimizer.step()
                    model.zero_grad()
                    epoch_total_loss += accu_loss.float()
                    # record time
                    time_end = time.time()
                    epoch_run_time += time_end - time_start
                    speed = epoch_run_time / (i_batch + 1)
                    eta = int((batch_cnt - (i_batch + 1)) * speed)
                    print(f'epoch:{i_epoch + 1} batch:{i_batch + 1} loss:{accu_loss.float()} avg loss:{epoch_total_loss / (i_batch + 1)} ETA:{eta // 3600}:{(eta % 3600) // 60}:{eta % 60}')
            else:
                model.zero_grad()
                outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                labels=masked_batch)
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                epoch_total_loss += loss.float()
                # record time
                time_end = time.time()
                epoch_run_time += time_end - time_start
                speed = epoch_run_time / (i_batch + 1)
                eta = int((batch_cnt - (i_batch + 1)) * speed)
                print(f'epoch:{i_epoch + 1} batch:{i_batch + 1} loss:{loss.float()} avg loss:{epoch_total_loss / (i_batch + 1)} ETA:{eta // 3600}:{(eta % 3600) // 60}:{eta % 60}')

            if (i_batch + 1) % save_batch_cnt == 0:
                save_name = '../' + model_path + '_ContTrain_epoch_' + str(i_epoch + 1) + '_batch_' + str(i_batch + 1) + '_bsz_' + str(mlm_bsz)
                print('saving models as:' + save_name)
                model.bert.save_pretrained(save_name)

        if (i_epoch + 1) % save_epoch_cnt == 0:
            save_name = '../' + model_path + '_ContTrain_epoch_' + str(i_epoch + 1) + '_bsz_' + str(mlm_bsz)
            print('saving models as:' + save_name)
            model.bert.save_pretrained(save_name)


if __name__ == '__main__':
    fire.Fire()
