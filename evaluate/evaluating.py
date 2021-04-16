import sys
sys.path.append('..')

from evaluate.eval_utils import *
from transformers import BertConfig, BertTokenizer
from tqdm import tqdm
import fire


def evaluate_event_detection():
    pass


def evaluate_trigger_extraction():
    pass


def evaluate_argument_extraction(aem_path, trigger_repr_path, repr_path):
    """
    models are stored at ../train, the prefix will be add in default
    :param aem_path:
    :param trigger_repr_path:
    :param repr_path:
    :return:
    """
    path_prefix = '../' if not inner_model else ''
    config = BertConfig.from_pretrained(path_prefix + model_path)
    # rfief = pickle.load(open('../train/rfief.pk', 'rb'))
    tokenizer = BertTokenizer.from_pretrained(path_prefix + model_path)

    aem = load_model_ae_aem('../train/' + aem_path, n_head, config.hidden_size, d_head, config.hidden_dropout_prob
                            , ltp_feature_cnt_fixed)
    trigger_repr_model = load_model_ae_trigger_repr('../train/' + trigger_repr_path, config.hidden_size)
    repr_model = load_model_ae_repr('../train/' + repr_path, model_path, config.hidden_size)
    role_mask = load_model_ae_role_mask('../train/rfief.pk')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    repr_model.to(device)
    trigger_repr_model.to(device)
    aem.to(device)

    val_sentences, val_types, val_triggers, arg_spans, val_syns = \
        pickle.load(open('../val_data_for_argument_extraction.pk', 'rb'))

    total, predict, correct = 0, 0, 0
    eval_result = []

    for i_val, val_sent in tqdm(list(enumerate(val_sentences))):
        val_type = val_types[i_val]
        val_trigger = val_triggers[i_val]
        val_syn = val_syns[i_val]
        val_span = arg_spans[i_val]

        h_styp = repr_model([val_sent], [val_type])
        h_styp, RPE = trigger_repr_model(h_styp, [val_trigger])
        start_logits, end_logits = aem(h_styp, val_syn.cuda(), RPE)  # (1, seq_l, len(role_types))
        start_logits, end_logits = role_mask(start_logits, [val_type]), role_mask(end_logits, [val_type])
        start_logits, end_logits = start_logits.squeeze().T, end_logits.squeeze().T  # (len(role_types), seq_l)
        start_results, end_results = (start_logits > argument_extraction_threshold).long().tolist() \
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
    eval_result.append((total, predict, correct, precision, recall, f_measure))
    open('eval_result.txt', 'a', encoding='utf-8').write(str(eval_result[-1]) + '\n')


if __name__ == '__main__':
    fire.Fire()