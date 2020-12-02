import modeling
import tokenization
import tensorflow as tf


class InputExample(object):
    def __init__(self, unique_id, text):
        self.unique_id = unique_id
        self.text = text


def read_examples(input_file):
    examples = []
    unique_id = 0
    with tf.gfile.GFile(input_file, "r") as reader:
        while True:
            line = tokenization.convert_to_unicode(reader.readline())
            if not line:
                break
            line = line.strip()
            examples.append(InputExample(unique_id, line))
            unique_id += 1

    return examples


if __name__ == '__main__':
    bert_config = modeling.BertConfig.from_json_file("../wwm_uncased_L-24_H-1024_A-16/bert_config.json")
