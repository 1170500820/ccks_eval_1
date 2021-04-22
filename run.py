import argparse
import pickle

from train.training import train_regular_model
from models.simple_ner import *
from transformers import AdamW
from settings import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # train ner model in default
    # parser.add_argument('--model', type=str)
    parser.add_argument('--PLM_path', type=str)

    parser.add_argument('--do_train', type=bool)
    parser.add_argument('--do_eval', type=bool)

    parser.add_argument('--lr', type=float)
    # parser.add_argument('--bsz', type=int)
    parser.add_argument('--epoch', type=int)

    args = parser.parse_args()
    print(args)

    lr = args.lr if args.lr else ner_lr
    epoch = args.epoch if args.epoch else ner_epoch
    plm_path = args.PLM_path if args.PLM_path else model_path

    # Step 1
    # Load model, optimizers.
    model = BertForTokenClassification.from_pretrained(plm_path)
    CELoss = nn.CrossEntropyLoss()

    # Step 2
    # Prepare data
    train_inputs, train_labels = pickle.load(open('train_input.pk')), pickle.load(open('train_labels.pk'))
    eval_inputs, eval_labels = pickle.load(open('val_input.pk')), pickle.load(open('val_labels.pk'))

    train_regular_model(
        lr=lr,
        epoch=epoch,
        save_freq=None,
        eval_freq=120,
        model=model,
        train_inputs=train_inputs,
        train_labels=train_labels,
        val_inputs=eval_inputs,
        val_labels=eval_labels,
        recorder=None
    )