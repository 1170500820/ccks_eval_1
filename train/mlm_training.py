# https://huggingface.co/transformers/model_doc/bert.html#bertformaskedlm

from transformers import BertTokenizer, BertForMaskedLM, AdamW
import torch
import fire
from settings import *
import pickle


def train_lm(lr=3e-5, epoch=20, mult_batch=True, mult_cnt=batch_mult_cnt):
    model = BertForMaskedLM.from_pretrained('../' + model_path)
    masked_ids, token_type_idss, attention_masks, input_idss = \
        pickle.load(open('../preprocess/train_data_for_mlm.pk', 'rb'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    for i_epoch in range(epoch):
        model.train()
        epoch_total_loss = 0.0
        temp_batch_cnt = 0
        accu_loss = None
        for i_batch, masked_batch in enumerate(masked_ids):
            masked_batch = masked_batch.cuda()
            input_ids = input_idss[i_batch].cuda()
            token_type_ids = token_type_idss[i_batch].cuda()
            attention_mask = attention_masks[i_batch].cuda()
            if mult_batch:
                outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                labels=masked_batch)
                if accu_loss is None:
                    accu_loss = outputs.loss
                else:
                    accu_loss = accu_loss + outputs.loss
                temp_batch_cnt += 1
                if temp_batch_cnt >= mult_cnt:
                    temp_batch_cnt = 0
                    accu_loss.backward()
                    optimizer.step()
                    model.zero_grad()
                    epoch_total_loss += accu_loss.float()
                    print(f'epoch:{i_epoch + 1} batch:{i_batch + 1} loss:{accu_loss.float()} avg loss:{epoch_total_loss / (i_batch + 1)}')
            else:
                model.zero_grad()
                outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                labels=masked_batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                epoch_total_loss += loss.float()
                print(f'epoch:{i_epoch + 1} batch:{i_batch + 1} loss:{loss.float()} avg loss:{epoch_total_loss / (i_batch + 1)}')


if __name__ == '__main__':
    fire.Fire()
