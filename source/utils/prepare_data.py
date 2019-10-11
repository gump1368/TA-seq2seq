import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset


def create_examples(data, max_length, set_type='text'):
    if set_type == 'text':
        assert len(data) <= max_length

        mask = [1]*len(data)
        padding = [0]*(max_length-len(data))

        text = data+padding
        mask = mask+padding

        return text, mask

    elif set_type == 'topic':
        assert len(data) <= max_length
        topics = []
        topics_mask = []

        for topic in data:
            text, mask = create_examples(topic, max_length, set_type='text')
            topics.append(text)
            topics_mask.append(mask)

        padding = [[0]*max_length]*(max_length-len(data))
        topics = topics+padding
        topics_mask = topics_mask+padding
        return topics, topics_mask

    else:
        return None, None


def create_features(data):
    src = []
    src_mask = []
    tgt = []
    tgt_mask = []
    topic = []
    topic_mask = []

    for line in data:
        try:
            _src, _src_mask = create_examples(line['src'], 120, 'text')
            _tgt, _tgt_mask = create_examples(line['tgt'], 120, 'text')
            _topic, _topic_mask = create_examples(line['cue'], 30, 'topic')
        except Exception as e:
            print(e)

        src.append(_src)
        src_mask.append(_src_mask)
        tgt.append(_tgt)
        tgt_mask.append(_tgt_mask)
        topic.append(_topic)
        topic_mask.append(_topic_mask)

    src = torch.LongTensor(src)
    src_mask = torch.LongTensor(src_mask)
    tgt = torch.LongTensor(tgt)
    tgt_mask = torch.LongTensor(tgt_mask)
    topic = torch.LongTensor(topic)
    topic_mask = torch.LongTensor(topic_mask)

    return TensorDataset(src, src_mask, tgt, tgt_mask, topic, topic_mask)

