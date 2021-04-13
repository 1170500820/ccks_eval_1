import sys
sys.path.append('..')


from settings import *
from transformers import AdamW, BertConfig
from models.event_detection import *
from models.sentence_representation_layer import *
from models.trigger_extraction_model import *
from models.argument_extraction_model import *
from models.role_mask import RoleMask
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


def train_trigger_extraction(repr_lr=2e-5, tem_lr = 1e-4, epoch=20, epoch_save_cnt=3, val=True, val_freq=300, val_start_epoch=-1):
    # Training Procedure Step - 1
    # Initiate model, model config, tokenizer, optimizer. Move models to devices
    config = BertConfig.from_pretrained('../' + model_path)
    tokenizer = BertTokenizer.from_pretrained('../' + model_path)

    # define models and optimizers
    repr_model = SentenceRepresentation('../' + model_path, config.hidden_size, pass_cln=False)
    optimizer_repr_plm = AdamW(repr_model.PLM.parameters(), lr=repr_lr)
    optimizer_repr_cln = AdamW(repr_model.CLN.parameters(), lr=tem_lr)
    tems, tem_optimizers = [], []
    for i in range(len(event_types_init)):
        tems.append(TriggerExtractionModel(n_head, config.hidden_size, d_head, config.hidden_dropout_prob, ltp_feature_cnt_fixed, pass_attn=False, pass_syn=False))
        tem_optimizers.append(AdamW(tems[-1].parameters(), lr=tem_lr))
    # tem = TriggerExtractionModel(n_head, config.hidden_size, d_head, config.hidden_dropout_prob, ltp_feature_cnt_fixed, pass_attn=False, pass_syn=False)
    # optimizer_tem = AdamW(tem.parameters(), lr=tem_lr)

    # training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    repr_model.to(device)
    for t in tems:
        t.to(device)

    # Training Procedure Step - 2
    # Prepare train and evaluate data (as batch)
    #   Both train and evaluate data should be iterable, every single data contains batch and necessary information
    train_sentences_batch, train_types_batch, train_gts_batch, train_syntactic_batch = \
        pickle.load(open('../train_data_for_trigger_extraction.pk', 'rb'))
    # train_sentences_batch, train_types_batch, train_gts_batch, train_syntactic_batch = \
    #     train_sentences_batch[3:], train_types_batch[3:], train_gts_batch[3:], train_syntactic_batch[3:]
    val_sentences, val_types, spans, val_syntactic_features = \
        pickle.load(open('../val_data_for_trigger_extraction.pk', 'rb'))
    eval_map = []

    # Training Procedure Step - 3
    # Start the training loop
    for i_epoch in range(epoch):
        # Step 3.1
        # Set model to train model before the epoch loop
        repr_model.train()
        for t in tems:
            t.train()
        # tem.train()
        epoch_total_loss = 0.0
        for i_batch, batch_ in enumerate(train_sentences_batch):
            # Step 3.1.1
            # Send the data into the model. run model.zero_grad() before sending the data
            # forward
            typ_batch = train_types_batch[i_batch]
            cur_type = typ_batch[0]
            cur_type_idx = event_types_init_index[cur_type]
            gt_batch = train_gts_batch[i_batch]
            syn_batch = train_syntactic_batch[i_batch]
            repr_model.zero_grad()
            for t in tems:
                t.zero_grad()
            # tem.zero_grad()
            h_styp = repr_model(batch_, typ_batch)
            start_logits, end_logits = tems[cur_type_idx](h_styp, syn_batch)

            # Step 3.1.2
            # Compute loss and backward
            loss = F.binary_cross_entropy(start_logits, gt_batch[0].cuda()) + F.binary_cross_entropy(end_logits, gt_batch[1].cuda())
            loss.backward()
            optimizer_repr_plm.step()
            optimizer_repr_cln.step()
            tem_optimizers[cur_type_idx].step()
            epoch_total_loss += loss.float()

            # Step 3.1.3
            # print train information
            print(f'epoch:{i_epoch + 1} batch:{i_batch + 1} loss:{loss.float()} epoch_avg_loss:{epoch_total_loss / (i_batch + 1)}')

            # Extra Step
            # Evaluation
            # Evaluation during training is convenience, because we dont have to repeat Step 1-2
            if (i_batch + 1) % val_freq == 0 and (i_epoch + 1) >= val_start_epoch:
                # Extra Step 1
                # Step 1-2 and run model.eval() for all models
                print('evaluating...')
                cur_eval_map = []
                for t in tems:
                    t.eval()
                repr_model.eval()
                results = []

                # Extra Step 2
                # Prepare the statistical variables
                total = 0
                predict = 0
                correct = 0
                left_part_correct = 0
                right_part_correct = 0
                word_correct = 0

                # Extra Step 3
                # Start the evaluating loop,
                for i_val, val_sent in list(enumerate(val_sentences)):
                    send_tokens = tokenizer.convert_ids_to_tokens(tokenizer(val_sent)['input_ids'])
                    cur_type_eval = val_types[i_val]
                    cur_type_idx_eval = event_types_init_index[cur_type_eval]
                    val_h_styp = repr_model(val_sent, val_types[i_val])
                    start_logits, end_logits = tems[cur_type_idx_eval](val_h_styp, val_syntactic_features[i_val])   # both (1, seq_l)
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
                    sorted_cur_span = list(cur_span)
                    sorted_cur_span.sort()
                    result_spans_set = set(map(tuple, result_spans))
                    sorted_result_span = list(result_spans_set)
                    sorted_result_span.sort()
                    total += len(cur_span)
                    predict += len(result_spans_set)
                    correct += len(cur_span.intersection(result_spans_set))
                    cur_eval_map.append(len(cur_span.intersection(result_spans_set)) / len(cur_span))
                    if cur_span != result_spans_set:
                        pass
                        #print('original sentence:', send_tokens)
                        #print('gt spans:', sorted_cur_span, '\n',  list(send_tokens[x[0] + 1: x[1] + 2] for x in sorted_cur_span))
                        #print('result spans:', sorted_result_span, '\n', list(send_tokens[x[0] + 1: x[1] + 2] for x in sorted_result_span))
                recall = correct / total if total != 0 else 0
                precision = correct / predict if predict != 0 else 0
                f_measure = (2 * recall * precision) / (recall + precision) if recall + precision != 0 else 0
                print(f'total:{total} predict:{predict}, correct:{correct}, precision:{precision}, recall:{recall}, f:{f_measure}')
                eval_map.append(cur_eval_map)
                # tem.train()
                for t in tems:
                    t.train()
                repr_model.train()
        pickle.dump(eval_map, open(f'eval_map_{i_epoch + 1}.pk', 'wb'))


def train_argument_extraction(repr_lr=2e-5, aem_lr=1e-4, epoch=50, epoch_save_cnt=3, val=True, val_freq=600, val_start_epoch=-1, inner_model=True):
    """

    :param repr_lr:
    :param aem_lr:
    :param epoch:
    :param epoch_save_cnt:
    :param val:
    :param val_freq:
    :param val_start_epoch:
    :param inner_model: 设置为True则代表使用内置的模型例如bert-base-chinese，这种情况下就不需要加路径前缀
    :return:
    """
    # Step 1
    # Initiate model, model config, tokenizer and optimizer, Move models to devices
    path_prefix = '../' if not inner_model else ''
    config = BertConfig.from_pretrained(path_prefix + model_path)
    rfief = pickle.load(open('rfief.pk', 'rb'))
    tokenizer = BertTokenizer.from_pretrained(path_prefix + model_path)
    #   define models and optimizers
    repr_model = SentenceRepresentation(path_prefix + model_path, config.hidden_size, pass_cln=False)
    trigger_repr_model = TriggeredSentenceRepresentation(config.hidden_size, pass_cln=False)
    aem = ArgumentExtractionModel(n_head, config.hidden_size, d_head, config.hidden_dropout_prob, ltp_feature_cnt_fixed)
    role_mask = RoleMask(rfief)
    optimizer_repr_plm = AdamW(repr_model.PLM.parameters(), lr=repr_lr)
    # optimizer_others = AdamW(list(repr_model.CLN.parameters()) + list(trigger_repr_model.parameters()) + list(aem.parameters()), lr=aem_lr)
    optimizer_repr_cln = AdamW(repr_model.CLN.parameters(), lr=aem_lr)
    optimizer_trigger_repr = AdamW(trigger_repr_model.parameters(), lr=aem_lr)
    optimizer_aem = AdamW(aem.parameters(), lr=aem_lr)
    #   devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    repr_model.to(device)
    trigger_repr_model.to(device)
    aem.to(device)

    # Step 2
    # Prepare train and evaluate data
    train_sentences_batch, train_types_batch, train_triggers_batch, train_gts_batch, train_syntactic_batch = \
        pickle.load(open('../train_data_for_argument_extraction.pk', 'rb'))
    val_sentences, val_types, val_triggers, arg_spans, val_syns = \
        pickle.load(open('../val_data_for_argument_extraction.pk', 'rb'))

    # # temp
    # start_cut, end_cut = 9, 15
    # train_sentences_batch, train_types_batch, train_triggers_batch, train_gts_batch, train_syntactic_batch \
    #     = train_sentences_batch[start_cut: end_cut], train_types_batch[start_cut: end_cut]\
    #     , train_triggers_batch[start_cut: end_cut], train_gts_batch[start_cut: end_cut]\
    #     , train_syntactic_batch[start_cut: end_cut]

    # Step 3
    # Start the training loop
    for i_epoch in range(epoch):
        repr_model.train()
        trigger_repr_model.train()
        aem.train()
        epoch_total_loss = 0.0
        for i_batch, batch_ in enumerate(train_sentences_batch):
            typ_batch = train_types_batch[i_batch]
            gt_batch = train_gts_batch[i_batch]
            syn_batch = train_syntactic_batch[i_batch]
            trigger_batch = train_triggers_batch[i_batch]
            # zero grad
            repr_model.zero_grad()
            trigger_repr_model.zero_grad()
            aem.zero_grad()

            h_styp = repr_model(batch_, typ_batch)
            h_styp, RPE = trigger_repr_model(h_styp, trigger_batch)
            start_logits, end_logits = aem(h_styp, syn_batch.cuda(), RPE)  # both (bsz, seq_l, len(role_types))
            start_logits_mask, end_logits_mask \
                = role_mask.return_weighted_mask(start_logits, typ_batch), role_mask.return_weighted_mask(end_logits, typ_batch)

            loss = F.binary_cross_entropy(start_logits, gt_batch[0].cuda(), start_logits_mask) + F.binary_cross_entropy(end_logits, gt_batch[1].cuda(), end_logits_mask)
            loss.backward()
            optimizer_repr_plm.step()
            # optimizer_others.step()
            optimizer_repr_cln.step()
            optimizer_trigger_repr.step()
            optimizer_aem.step()    # 不同optimizer分别step和一个optimizer自己step，结果有区别吗

            epoch_total_loss += loss.float()
            print(f'epoch:{i_epoch + 1} batch:{i_batch + 1} loss:{loss.float()} epoch_avg_loss:{epoch_total_loss / (i_batch + 1)}')

            if (i_batch + 1) % val_freq == 0 and (i_epoch + 1) >= val_start_epoch:
                print('evaluating')
                repr_model.eval()
                trigger_repr_model.eval()
                aem.eval()

                total, predict, correct = 0, 0, 0

                for i_val, val_sent in tqdm(list(enumerate(val_sentences))):
                    val_type = val_types[i_val]
                    val_trigger = val_triggers[i_val]
                    val_syn = val_syns[i_val]
                    val_span = arg_spans[i_val]

                    h_styp = repr_model([val_sent], [val_type])
                    h_styp, RPE = trigger_repr_model(h_styp, [val_trigger])
                    start_logits, end_logits = aem(h_styp, val_syn.cuda(), RPE) # (1, seq_l, len(role_types))
                    start_logits, end_logits = role_mask(start_logits, [val_type]), role_mask(end_logits, [val_type])
                    start_logits, end_logits = start_logits.squeeze().T, end_logits.squeeze().T # (len(role_types), seq_l)
                    start_results, end_results = (start_logits > argument_extraction_threshold).long().tolist()\
                        , (end_logits > argument_extraction_threshold).long().tolist()
                    result_spans = []
                    for i_span in range(len(role_types)):
                        result_span = argument_span_determination(start_results[i_span], end_results[i_span]
                                                                  , start_logits[i_span], end_logits[i_span])
                        result_spans.append(result_span)

                    for i_compare in range(len(role_types)):
                        total += len(val_span[i_compare])
                        predict += len(result_spans[i_compare])
                        correct += len(set(map(tuple, val_span[i_compare])).intersection(set(map(tuple, result_spans[i_compare]))))

                recall = correct / total if total != 0 else 0
                precision = correct / predict if predict != 0 else 0
                f_measure = (2 * recall * precision) / (recall + precision) if recall + precision != 0 else 0
                print(
                    f'total:{total} predict:{predict}, correct:{correct}, precision:{precision}, recall:{recall}, f:{f_measure}')


if __name__ == '__main__':
    fire.Fire()

