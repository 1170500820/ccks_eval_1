import re
import json


def read_file(filepath='../data/train_base.json'):
    """
    简单读取文件并转化为list
    :param filepath:
    :return:
    """
    return list(map(json.loads, open(filepath, 'r', encoding='utf-8').read().strip().split('\n')))


if __name__ == '__main__':
    percentage_regex = r'\d+(\.\d+)?%'
    digits_regex = r'(不)?(低于|超过|超)?\d+(\.\d+)?(万|亿)?(股)?'
    date_regex = r'((\d+|今|去|前)\s?年底?)?(\d+月份?)?(\d+日)?'
    data = read_file()
    sentences = list(map(lambda x: x['content'], data))
    digit_role = {'amount', 'number', 'share-org', 'date', 'proportion', 'share-per', 'money'}
    digit_spans = []
    for d in data:
        digit_spans.append([])
        for e in d['events']:
            for r in e['mentions']:
                if r['role'] in digit_role:
                    digit_spans[-1].append(r)

