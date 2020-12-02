import numpy as np
import random
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer, BertForMultipleChoice
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange


class Example(object):
    # 确定样本格式
    def __init__(self, data_id, statement1, statement2, label=None):
        self.data_id = data_id
        self.statements = [statement1, statement2]
        self.label = label


def read_examples(input_file, have_answer=True):
    with open(input_file, "r", encoding="utf-8") as f:
        examples = []
        for line in f.readlines():
            newline = eval(line)
            if have_answer:
                label = newline['correct']
            else:
                label = 0
            examples.append(Example(data_id=newline["data_id"], statement1=newline["statement1"],
                                    statement2=newline["statement2"], label=label))
    return examples


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [{'input_ids': input_ids} for _, input_ids in choices_features]
        self.label = label


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    # - [CLS] statement1 [SEP]
    # - [CLS] statement2 [SEP]
    features = []
    for example_index, example in enumerate(examples):

        choices_features = []
        for index, statement in enumerate(example.statements):
            if '/' in statement:
                tmp = statement.split('/')
                statement = ['[CLS]'] + tokenizer.tokenize(tmp[0]) + ['[SEP]'] + tokenizer.tokenize(tmp[1]) + ['[SEP]']
            else:
                tokens = ['[CLS]'] + tokenizer.tokenize(statement) + ['[SEP]']
            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
                tokens[-1] = '[SEP]'
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids = input_ids + padding
            choices_features.append((tokens, input_ids))

        label = example.label

        features.append(InputFeatures(example_id=example.data_id, choices_features=choices_features, label=label))

    return features


def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]


def get_eval_dataloader(eval_features, eval_batch_size):
    all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_label)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
    return eval_dataloader


def get_train_dataloader(train_features, train_batch_size):
    all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_label)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
    return train_dataloader


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def evaluate(model, device, eval_dataloader, desc="Train"):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    eval_logits = []
    eval_outputs = []
    for input_ids, label_ids in tqdm(eval_dataloader, desc=desc):
        input_ids = input_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss, tmp_eval_logits = model(input_ids=input_ids, labels=label_ids)[:2]

        tmp_eval_logits = tmp_eval_logits.detach().cpu().numpy()
        eval_logits = eval_logits + list(tmp_eval_logits.tolist())
        outputs = np.argmax(tmp_eval_logits, axis=1)
        eval_outputs = eval_outputs + list(outputs)
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(tmp_eval_logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    # """
    if desc == "Dev":
        with open("../result/bert_finetuning.csv", "w") as f_pred:
            with open('../dataset/test_bert.txt', "r", encoding="utf-8") as fin:
                for i, line in enumerate(fin.readlines()):
                    newline = eval(line)
                    label_pred = eval_outputs[i]
                    f_pred.write(str(newline["data_id"]) + "," + str(label_pred) + "\n")

        with open("../result/bert_finetuning_logits.txt", "w", encoding="utf_8") as f_logit:
            with open('../dataset/test_bert.txt', "r", encoding="utf-8") as fin:
                for i, line in enumerate(fin.readlines()):
                    newline = eval(line)
                    logit = eval_logits[i]
                    f_logit.write(str({"data_id": newline["data_id"], "logit": logit}) + "\n")
    # """
    return eval_loss, eval_accuracy


hyperparameters = {
                    "max_sent_length": 70, "train_batch_size": 32, "eval_batch_size": 32,
                    "learning_rate": 3e-5, "num_epoch": 100, "seed": 114,
                    "patience": 4, "gradient_accumulation_steps": 4
                    }


def train():
    # 检查配置，获取超参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device:{} n_gpu:{}".format(device, n_gpu))
    seed = hyperparameters["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    max_seq_length = hyperparameters["max_sent_length"]
    gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
    num_epochs = hyperparameters["num_epoch"]
    train_batch_size = hyperparameters["train_batch_size"] // hyperparameters["gradient_accumulation_steps"]
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True)
    model = BertForMultipleChoice.from_pretrained("bert-large-uncased")
    model.to(device)

    # 优化器
    param_optimizer = list(model.named_parameters())

    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 载入数据
    train_examples = read_examples('../dataset/train_bert.txt')
    dev_examples = read_examples('../dataset/test_bert.txt')
    nTrain = len(train_examples)
    nDev = len(dev_examples)
    num_train_optimization_steps = int(nTrain / train_batch_size / gradient_accumulation_steps) * num_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=hyperparameters["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_train_optimization_steps),
                                                num_training_steps=num_train_optimization_steps)

    global_step = 0
    train_features = convert_examples_to_features(train_examples, tokenizer, max_seq_length)
    dev_features = convert_examples_to_features(dev_examples, tokenizer, max_seq_length)
    train_dataloader = get_train_dataloader(train_features, train_batch_size)
    dev_dataloader = get_eval_dataloader(dev_features, hyperparameters["eval_batch_size"])
    print("Num of train features:{}".format(nTrain))
    print("Num of dev features:{}".format(nDev))
    best_dev_accuracy = 0
    best_dev_epoch = 0
    no_up = 0

    epoch_tqdm = trange(int(num_epochs), desc="Epoch")
    for epoch in epoch_tqdm:
        model.train()

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, label_ids = batch
            loss, logits = model(input_ids=input_ids, labels=label_ids)[:2]
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        train_loss, train_accuracy = evaluate(model, device, train_dataloader, "Train")
        dev_loss, dev_accuracy = evaluate(model, device, dev_dataloader, "Dev")

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_dev_epoch = epoch + 1
            no_up = 0

        else:
            no_up += 1
        tqdm.write("\t ***** Eval results (Epoch %s) *****" % str(epoch + 1))
        tqdm.write("\t train_accuracy = %s" % str(train_accuracy))
        tqdm.write("\t dev_accuracy = %s" % str(dev_accuracy))
        tqdm.write("")
        tqdm.write("\t best_dev_accuracy = %s" % str(best_dev_accuracy))
        tqdm.write("\t best_dev_epoch = %s" % str(best_dev_epoch))
        tqdm.write("\t no_up = %s" % str(no_up))
        tqdm.write("")
        if no_up >= hyperparameters["patience"]:
            epoch_tqdm.close()
            break


if __name__ == "__main__":
    train()
