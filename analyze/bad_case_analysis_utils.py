from settings import role_types, model_path, inner_model, ner_threshold
import numpy as np
import pickle
from transformers import BertTokenizer
from evaluate.eval_utils import argument_span_determination
import torch



def score_argument_extraction(sentence: str, sentence_types: str, predict_spans: [[(int, int), ], ], ground_truth_spans: [[(int, int)], ]):
    """
    为在一个句子Sentence上的论元预测进行评分
    在最初的实现，直接使用f1值进行评判
    如果有不同的侧重点，比如说
    :param sentence:
    :param sentence_types: 句子所对应的事件类型
    :param predict_spans: 预测出的span [[(int, int), ], ]
    :param ground_truth_spans: gt span [[(int, int), ], ]
    :return: [0, 1] 0代表最差，1代表最好
    """
    total, predict, correct = 0, 0, 0
    assert len(predict_spans) == len(ground_truth_spans) == len(role_types)
    for i, ps in enumerate(predict_spans):
        gt = ground_truth_spans[i]
        ps_set, gt_set = set(ps), set(gt)
        total += len(gt_set)
        predict += len(ps_set)
        correct += len(ps_set.intersection(gt_set))
    recall = correct / total if total != 0 else 0
    precision = correct / predict if predict != 0 else 0
    f_measure = (2 * recall * precision) / (recall + precision) if recall + precision != 0 else 0
    if total == predict == 0:
        # 此时预测是正确的
        return 1.0
    else:
        return f_measure


def show_argument_extraction_badcase():
    pass


# def show(number, recent=10):
#     tokens = tokenizer.convert_ids_to_tokens(tokenizer(val_sentences[number])['input_ids'])[1:]
#     gts = arg_spans[number]
#     gt_words = []
#     for gt in gts:
#         gt_word = list(map(lambda x: (x, ''.join(tokens[x[0]: x[1] + 1])), gt))
#         gt_words.append(gt_word)
#     results = []
#     for i in range(recent):
#         results.append([])
#         for j in range(len(gt_words)):
#             results[-1].append(record.result_spans[39 - i][number][j])
#     print(f'origin sentence:{val_sentences[number]}')
#     print(f'Tokens:{tokens}')
#     print(f'Type:{val_types[number]}')
#     print(f'Trigger:{(val_triggers[number], tokens[val_triggers[number][0]: val_triggers[number][1] + 1])}')
#     print(f'gt span:{gt_words}')
#     for i in range(recent):
#         ws = []
#         for k in range(len(gt_words)):
#             w = list(map(lambda x: (x, ''.join(tokens[x[0]: x[1] + 1])), results[i][k]))
#             ws.append(w)
#         print(f'result {i + 1}:{ws}')

def get_total_find_f1(gt: [[], ], result: [[], ]):
    """
    忽略实体类别，仅考虑找出来的f1
    :param gt:
    :param result:
    :return:
    """
    gt_sets, result_sets = set(), set()
    for g in gt:
        for sp in g:
            gt_sets.add(sp)
    for r in result:
        for sp in r:
            result_sets.add(sp)
    total = len(gt_sets)
    predict = len(result_sets)
    correct = len(gt_sets.intersection(result_sets))
    precision = correct / predict if predict != 0 else 0
    recall = correct / total if total != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    if total == predict == 0:
        f1 = 1
    return f1


class EvalRecord:
    def __init__(self, ground_truth_spans):
        self.scores = []
        self.result_spans = []
        self.measures = []
        self.precisions, self.recalls, self.f1s = [], [], []
        self.ground_truth_spans = ground_truth_spans
        self.avg_scores = []
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

        self.sentences, self.contains_sentences = [], False
        self.types, self.contains_types = [], False
        self.trigger_span, self.contains_trigger_span = [], False

        self.val_sentences, self.val_types, self.val_triggers, self.arg_spans, self.val_syns = [], [], [], [], []
        self.load_val_data()

    def add_sentences(self, sentences: [str, ]):
        self.sentences = sentences
        self.contains_sentences = True

    def add_types(self, types: [str, ]):
        self.types = types
        self.contains_types = True

    def add_trigger_span(self, trigger_spans: [(int, int), ]):
        self.trigger_span = trigger_spans
        self.contains_trigger_span = True

    def record(self, timestep, score, predict_span):
        """
        记录一次evaluate中，某一个sample的预测结果。
        predict_span和ground_truth_span将以节省内存的形式存储。
        :param timestep: 从0开始
        :param score:
        :param predict_span:
        :param ground_truth_span:
        :return:
        """
        if len(self.scores) < timestep + 1:
            self.scores.append([])
            self.result_spans.append([])
        self.scores[-1].append(score)
        self.result_spans[-1].append(predict_span)

    def record_measures(self, timestep, total, predict, correct):
        self.measures.append((total, predict, correct))
        recall = correct / total if total != 0 else 0
        precision = correct / predict if predict != 0 else 0
        f_measure = (2 * recall * precision) / (recall + precision) if recall + precision != 0 else 0
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1s.append(f_measure)

    def inquire_gt(self, sample_idx):
        """
        查询某一个sample的gt
        :param sample_idx:
        :return:
        """
        return self.ground_truth_spans[sample_idx]

    def inquire_predict(self, sample_idx, timestep):
        """
        查询某一个timestep下，某一个sample的预测结果
        :param sample_idx:
        :param timestep:
        :return:
        """
        if len(self.result_spans) >= timestep:
            raise Exception('no such timestep!')
        if sample_idx >= len(self.ground_truth_spans):
            raise Exception('no such sample idx!')
        return self.result_spans[timestep][sample_idx]

    def analyze(self):
        # calculate avg first
        np_scores = np.array(self.scores)
        avgs = np_scores.mean(axis=1)

    def ranklist(self):
        return np.array(self.scores).mean(axis=0).argsort().tolist()

    def scorelist(self):
        """
        return sorted score
        :return:
        """
        return np.array(self.scores).mean(axis=0).sort().tolist()

    def summarize(self, num=20) -> str:
        """
        输出总结信息
        :return:
        """
        f1_np = np.array(self.f1s[-num: ])
        max_f1 = max(self.f1s)
        mean_f1 = f1_np.mean()
        string = f'最大f1:{max_f1}\n后{num}次评价的平均f1{mean_f1}'
        return string


    def save(self, filename):
        pickle.dump([self.scores, self.measures, self.result_spans, self.ground_truth_spans], open(filename + '.' + str(len(self.measures)), 'wb'))

    @staticmethod
    def from_file(filename):
        scores, measure, result_spans, ground_truth_spans = pickle.load(open(filename, 'rb'))
        r = EvalRecord(ground_truth_spans)
        r.scores = scores
        r.result_spans = result_spans
        r.measures = measure
        return r

    def load_val_data(self):
        self.val_sentences, self.val_types, self.val_triggers, self.arg_spans, self.val_syns = \
            pickle.load(open('../val_data_for_argument_extraction.pk', 'rb'))

    def show(self, number, recent=15):
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer(self.val_sentences[number])['input_ids'])[1:]
        gts = self.arg_spans[number]
        gt_words = []
        for gt in gts:
            gt_word = list(map(lambda x: (x, ''.join(tokens[x[0]: x[1] + 1])), gt))
            gt_words.append(gt_word)
        results = []
        for i in range(recent):
            results.append([])
            for j in range(len(gt_words)):
                results[-1].append(self.result_spans[39 - i][number][j])
        print(f'origin sentence:{self.val_sentences[number]}')
        print(f'Tokens:{tokens}')
        print(f'Type:{self.val_types[number]}')
        print(f'Trigger:{(self.val_triggers[number], tokens[self.val_triggers[number][0]: self.val_triggers[number][1] + 1])}')
        print(f'gt span:{gt_words}')
        for i in range(recent):
            ws = []
            for k in range(len(gt_words)):
                w = list(map(lambda x: (x, ''.join(tokens[x[0]: x[1] + 1])), results[i][k]))
                ws.append(w)
            print(f'result {i + 1}:{ws}')


class NerEvalRecorder:
    def __init__(self, gt):
        self.span = []
        self.measures = []  # 存放评价指标
        self.gt = gt

    def record(self, spans: [], i_epoch, i_batch):
        self.span.append(spans)
        self.measures.append(None)

    @staticmethod
    def logits2span(results=None):
        """
        在EvalRecorder父类中，应该是logits2label
        统一的是将模型的输出转化为label容易分辨的格式
        最好与从数据中直接读取的格式相同且足够简单
        这样就能够防止label格式过多而难以管理
        :param results: [(1, seq_l, 1), (1, seq_l, 1)]
        :return:
        """
        starts, ends = torch.sigmoid(results[0].squeeze()), torch.sigmoid(results[1].squeeze())
        start_results, end_results = (starts > ner_threshold).long().tolist(), (ends > ner_threshold).long().tolist()
        span = argument_span_determination(start_results, end_results, starts, ends)
        return list(map(tuple, span))

    @staticmethod
    def convert(logits=None):
        return NerEvalRecorder.logits2span(logits=logits)

    @staticmethod
    def score(predict, gt):
        pass

    def evaluate(self):
        """
        evaluate self.span[-1] and self.gt
        :return:
        """
        total, predict, correct = 0, 0, 0
        for i, g in enumerate(self.gt):
            s = self.span[-1][i]
            gt_set, result_set = set(g), set(s)
            total += len(gt_set)
            predict += len(result_set)
            correct += len(gt_set.intersection(result_set))

        precision = correct / predict if predict != 0 else 0
        recall = correct / total if total != 0 else 0
        f = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        s = f"total:{total}, predict:{predict}, correct:{correct}, precision:{precision}, recall:{recall}, f1:{f}"
        self.measures[-1] = (precision, recall, f)
        return (precision, recall, f), s
