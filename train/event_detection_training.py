import sys
sys.path.append('..')


from settings import *
from transformers import AdamW, BertConfig
from models.event_detection import *
from models.sentence_representation_layer import *
from models.trigger_extraction_model import *
from evaluate.eval_utils import *
import torch
import torch.nn.functional as F
import pickle
import fire
from tqdm import tqdm


def train_event_detection(lr=3e-5, epoch=20, epoch_save_cnt=3, val=True, val_freq=300):
    config = BertConfig.from_pretrained(model_path)
    config.num_labels = len(event_types_init)
    config.pretrain_path = model_path
    model = EventDetection(config)
    optimizer = AdamW(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    sentence_batch = pickle.load(open('../sentences.pk', 'rb'))
    gt_batch = pickle.load(open('../gts.pk', 'rb'))

    val_sentences = pickle.load(open('../val_sentences_tokenized.pk', 'rb'))
    val_gts = pickle.load(open('../val_gts.pk', 'rb'))

    for i_epoch in range(epoch):
        model.train()
        epoch_total_loss = 0.0
        for i_batch, batch_ in enumerate(sentence_batch):
            cur_gt = gt_batch[i_batch].cuda()
            input_ids = batch_['input_ids']
            token_type_ids = batch_['token_type_ids']
            attention_mask = batch_['attention_mask']
            input_ids = input_ids.cuda()
            token_type_ids = token_type_ids.cuda()
            attention_mask = attention_mask.cuda()
            model.zero_grad()
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            reshpaed_outputs = outputs.squeeze()
            loss = F.binary_cross_entropy_with_logits(reshpaed_outputs, cur_gt)
            loss.backward()
            optimizer.step()
            epoch_total_loss += loss.float()

            print(f'epoch:{i_epoch + 1} batch:{i_batch + 1} loss:{loss.float()} epoch_avg_loss:{epoch_total_loss / (i_batch + 1)}')

            if (i_batch + 1) % val_freq  == 0:
                # eval
                print('evaluating...')
                model.eval()
                results = []
                for test_sent in val_sentences:
                    input_ids = test_sent['input_ids'].cuda()
                    token_type_ids = test_sent['token_type_ids'].cuda()
                    attention_mask = test_sent['attention_mask'].cuda()
                    result = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                    # result should be (1, 1, num_labels)
                    # after squeezing (num_labels)
                    # results appended a list of size num_labels
                    results.append(torch.gt(result.squeeze(), event_detection_theshold).long().tolist())

                total_cnt = len(val_gts)
                fully_correct = 0
                for i_val in range(total_cnt):
                    if val_gts[i_val] == results[i_val]:
                        fully_correct += 1
                fully_correct_p = fully_correct / total_cnt

                total, predict, correct = 0, 0, 0
                for i_val in range(total_cnt):
                    total += val_gts[i_val].count(1)
                    predict += results[i_val].count(1)
                    for k in range(len(val_gts[i_val])):
                        if val_gts[i_val][k] == results[i_val][k] == 1:
                            correct += 1
                precision = correct / predict if predict != 0 else 0
                recall = correct / total if total != 0 else 0
                f = (2 * precision * recall) / (precision + recall) if precision != 0 and recall != 0 else 0
                print(f'完全一致率:{fully_correct_p} precision:{precision} recall:{recall} f:{f}\n'
                      f'fully_correct_cnt:{fully_correct} total:{total} predict:{predict} correct:{correct}')
                model.train()


def train_trigger_extraction(repr_lr=2e-5, tem_lr = 1e-4, epoch=20, epoch_save_cnt=3, val=True, val_freq=300):
    config = BertConfig.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # define models and optimizers
    repr_model = SentenceRepresentation(model_path, config.hidden_size, pass_cln=False)
    tem = TriggerExtractionModel(n_head, config.hidden_size, d_head, config.hidden_dropout_prob, ltp_feature_cnt_fixed, pass_attn=False, pass_syn=False)
    optimizer_repr_plm = AdamW(repr_model.PLM.parameters(), lr=repr_lr)
    optimizer_repr_cln = AdamW(repr_model.CLN.parameters(), lr=tem_lr)
    optimizer_tem = AdamW(tem.parameters(), lr=tem_lr)

    # training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    repr_model.to(device)
    tem.to(device)

    # prepare data
    train_sentences_batch, train_types_batch, train_gts_batch, train_syntactic_batch = \
        pickle.load(open('../train_data_for_trigger_extraction.pk', 'rb'))
    # train_sentences_batch, train_types_batch, train_gts_batch, train_syntactic_batch = \
    #     train_sentences_batch[3:], train_types_batch[3:], train_gts_batch[3:], train_syntactic_batch[3:]
    val_sentences, val_types, spans, val_syntactic_features = \
        pickle.load(open('../val_data_for_trigger_extraction.pk', 'rb'))
    for i_epoch in range(epoch):
        repr_model.train()
        tem.train()
        epoch_total_loss = 0.0
        for i_batch, batch_ in enumerate(train_sentences_batch):
            # forward
            typ_batch = train_types_batch[i_batch]
            gt_batch = train_gts_batch[i_batch]
            syn_batch = train_syntactic_batch[i_batch]
            repr_model.zero_grad()
            tem.zero_grad()
            h_styp = repr_model(batch_, typ_batch)
            start_logits, end_logits = tem(h_styp, syn_batch)

            # compute loss and backward
            loss = F.binary_cross_entropy(start_logits, gt_batch[0].cuda()) + F.binary_cross_entropy(end_logits, gt_batch[1].cuda())
            loss.backward()
            optimizer_repr_plm.step()
            optimizer_repr_cln.step()
            optimizer_tem.step()
            epoch_total_loss += loss.float()

            print(f'epoch:{i_epoch + 1} batch:{i_batch + 1} loss:{loss.float()} epoch_avg_loss:{epoch_total_loss / (i_batch + 1)}')

            # eval
            if (i_batch + 1) % val_freq == 0:
                print('evaluating...')
                tem.eval()
                repr_model.eval()
                results = []

                total = 0
                predict = 0
                correct = 0
                left_part_correct = 0
                right_part_correct = 0
                word_correct = 0
                for i_val, val_sent in list(enumerate(val_sentences)):
                    send_tokens = tokenizer.convert_ids_to_tokens(tokenizer(val_sent)['input_ids'])
                    val_h_styp = repr_model(val_sent, val_types[i_val])
                    start_logits, end_logits = tem(val_h_styp, val_syntactic_features[i_val])   # both (1, seq_l)
                    # print('starts:', start_logits)
                    # print('start\'s size:', start_logits.size())
                    # print('ends:', end_logits)
                    start_logits = start_logits.squeeze()   # (seq_l)
                    end_logits = end_logits.squeeze()   # (seq_l)
                    # print('start after squeeze:', start_logits)
                    # print('end after squeeze:', end_logits)
                    binary_starts = (start_logits >= trigger_extraction_threshold).long()
                    binary_ends = (end_logits >= trigger_extraction_threshold).long()
                    # print('binary starts:', binary_starts)
                    # print('binary ends:', binary_ends)
                    # start, end = start_logits.argmax(dim=0), end_logits.argmax(dim=0)
                    result_spans = argument_span_determination(binary_starts, binary_ends, start_logits, end_logits)
                    cur_span = set(map(lambda x: (x[0], x[1] - 1), spans[i_val])) # {(start, end), ...}
                    result_spans_set = set(map(tuple, result_spans))
                    total += len(cur_span)
                    predict += len(result_spans_set)
                    correct += len(cur_span.intersection(result_spans_set))
                recall = correct / total if total != 0 else 0
                precision = correct / predict if predict != 0 else 0
                f_measure = (2 * recall * precision) / (recall + precision) if recall + precision != 0 else 0
                print(f'total:{total} predict:{predict}, correct:{correct}, precision:{precision}, recall:{recall}, f:{f_measure}')
                tem.train()
                repr_model.train()


if __name__ == '__main__':
    fire.Fire()

