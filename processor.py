# -*- coding:utf-8 -*-

import torch
import os
from openprompt.data_utils.utils import InputExample
from dataset.KGWebOfScience.trans_format import get_mapping
from tqdm import tqdm
from dataset.rcv1.trans_format import get_mapping as rcv1_get_mapping
from dataset.WebOfScience.my_dataset import sub_dataset as wos_sub_dataset
from dataset.rcv1.my_dataset import sub_dataset as rcv1_sub_dataset
from dataset.DBpedia.trans_format import get_mapping
from dataset.DBpedia.my_dataset import sub_dataset
base_path = "./"
from dataset.KGWebOfScience.my_dataset import sub_dataset

base_path = "./"


class WOSProcessor:

    def __init__(self, ratio=-1, seed=171, shot=16, ratio_flag=0):
        super().__init__()
        self.name = 'KGWebOfScience'
        label0_list, label1_list, label0_label2id, label1_label2id, label0_to_label1_mapping, label1_to_label0_mapping = get_mapping()
        self.labels = label1_list
        self.coarse_labels = label0_list
        self.all_labels = label0_list + label1_list
        self.label_list = [label0_list, label1_list]
        self.label0_to_label1_mapping = label0_to_label1_mapping
        self.label1_to_label0_mapping = label1_to_label0_mapping
        self.label0_label2id=label0_label2id
        self.label1_label2id=label1_label2id

        self.data_path = os.path.join(base_path, "dataset", "KGWebOfScience")
        self.flat_slot2value, self.value2slot, self.depth2label = self.get_tree_info()
        self.hier_mapping = [[label0_to_label1_mapping, label1_to_label0_mapping]]
        
        self.ratio = ratio
        self.seed = seed
        self.shot = shot
        self.dataset = sub_dataset(self.shot, self.seed, self.ratio, ratio_flag=ratio_flag)
        print("length dataset['train']:", len(self.dataset['train']))

        self.train_data = self.get_dataset("train")

        self.dev_data = self.get_dataset("val")
        self.test_data = self.get_dataset("test")
        self.train_example = self.convert_data_to_examples(self.train_data)
        self.dev_example = self.convert_data_to_examples(self.dev_data)
        self.test_example = self.convert_data_to_examples(self.test_data)

        self.train_inputs = [i[0] for i in self.train_data]
        self.dev_inputs = [i[0] for i in self.dev_data]
        self.test_inputs = [i[0] for i in self.test_data]

        self.size = len(self.train_example) + len(self.test_example)

    def get_tree_info(self):
        flat_slot2value = torch.load(os.path.join(self.data_path, 'slot.pt'))

        value2slot = {}
        num_class = 0
        for s in flat_slot2value:
            for v in flat_slot2value[s]:
                value2slot[v] = s
                if num_class < v:
                    num_class = v
        num_class += 1
        for i in range(num_class):
            if i not in value2slot:
                value2slot[i] = -1

        def get_depth(x):
            depth = 0
            while value2slot[x] != -1:
                depth += 1
                x = value2slot[x]
            return depth

        depth_dict = {i: get_depth(i) for i in range(num_class)}
        max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
        depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}
        return flat_slot2value, value2slot, depth2label

    def get_dataset(self, type="train"):
        data = []
        cur_dataset = self.dataset[type]
        length = len(cur_dataset)
        for i in tqdm(range(length)):
            text_a = cur_dataset[i][0]
            label = cur_dataset[i][1]
            data.append([text_a,label])
        return data

    def convert_data_to_examples(self, data):
        examples = []
        for idx, sub_data in enumerate(data):
            examples.append(InputExample(guid=str(idx), text_a=sub_data[0], label=sub_data[1]))

        return examples

class KGWOSProcessor:
    def __init__(self, ratio=-1, seed=171, shot=16, ratio_flag=0):
        super().__init__()
        self.name = 'KGWebOfScience'
        label0_list, label1_list, label0_label2id, label1_label2id, label0_to_label1_mapping, label1_to_label0_mapping = get_mapping()
        self.labels = label1_list
        self.coarse_labels = label0_list
        self.all_labels = label0_list + label1_list
        self.label_list = [label0_list, label1_list]
        self.label0_to_label1_mapping = label0_to_label1_mapping
        self.label1_to_label0_mapping = label1_to_label0_mapping
        self.label0_label2id=label0_label2id
        self.label1_label2id=label1_label2id

        self.data_path = os.path.join(base_path, "dataset", "KGWebOfScience")
        self.flat_slot2value, self.value2slot, self.depth2label = self.get_tree_info()
        self.hier_mapping = [[label0_to_label1_mapping, label1_to_label0_mapping]]
        
        self.ratio = ratio
        self.seed = seed
        self.shot = shot
        self.dataset = sub_dataset(self.shot, self.seed, self.ratio, ratio_flag=ratio_flag)
        print("length dataset['train']:", len(self.dataset['train']))

        self.train_data = self.get_dataset("train")

        self.dev_data = self.get_dataset("val")
        self.test_data = self.get_dataset("test")
        self.train_example,self.train_entities = self.convert_data_to_examples(self.train_data)
        self.dev_example,self.dev_entities = self.convert_data_to_examples(self.dev_data)
        self.test_example,self.test_entities = self.convert_data_to_examples(self.test_data)

        self.train_inputs = [i[0] for i in self.train_data]
        self.dev_inputs = [i[0] for i in self.dev_data]
        self.test_inputs = [i[0] for i in self.test_data]

        self.size = len(self.train_example) + len(self.test_example)

    def get_tree_info(self):
        flat_slot2value = torch.load(os.path.join(self.data_path, 'slot.pt'))

        value2slot = {}
        num_class = 0
        for s in flat_slot2value:
            for v in flat_slot2value[s]:
                value2slot[v] = s
                if num_class < v:
                    num_class = v
        num_class += 1
        for i in range(num_class):
            if i not in value2slot:
                value2slot[i] = -1

        def get_depth(x):
            depth = 0
            while value2slot[x] != -1:
                depth += 1
                x = value2slot[x]
            return depth

        depth_dict = {i: get_depth(i) for i in range(num_class)}
        max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
        depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}
        return flat_slot2value, value2slot, depth2label

    def get_dataset(self, type="train"):
        data = []
        cur_dataset = self.dataset[type]
        length = len(cur_dataset)
        for i in tqdm(range(length)):
            text_a = cur_dataset[i][0]
            label = cur_dataset[i][1]
            entity = cur_dataset[i][2]

            data.append([text_a,label,entity])
        return data

    def convert_data_to_examples(self, data):
        examples = []
        entities=[]
        for idx, sub_data in enumerate(data):
            examples.append(InputExample(guid=str(idx), text_a=sub_data[0], label=sub_data[1]))
            entities.append(sub_data[2])

        return examples,entities
class RCV1Processor:

    def __init__(self,ratio=-1, seed=171, shot=-1, ratio_flag=0):
        self.name = 'rcv1'
        label0_list, label1_list,label2_list,label3_list, label0_label2id, label1_label2id, label2_label2id,label3_label2id,label0_to_label1_mapping, label1_to_label0_mapping,label1_to_label2_mapping,label2_to_label1_mapping,label2_to_label3_mapping,label3_to_label2_mapping = rcv1_get_mapping()
        self.labels=[label2_list]
        self.coarese_labels = []
        self.all_labels = []
        self.label0_to_label1_mapping = label0_to_label1_mapping
        self.label1_to_label0_mapping = label1_to_label0_mapping
        self.label1_to_label2_mapping = label1_to_label2_mapping
        self.label2_to_label1_mapping = label2_to_label1_mapping
        self.label2_to_label3_mapping = label2_to_label3_mapping
        self.label3_to_label2_mapping = label3_to_label2_mapping
        self.description=os.path.join(base_path,"dataset","rcv1","rcv1_label_description_10.json")
        self.data_path = os.path.join(base_path, "dataset", "rcv1")
        self.flat_slot2value, self.value2slot, self.depth2label = self.get_tree_info()
        self.hier_mapping = [[label0_to_label1_mapping, label1_to_label0_mapping],[label1_to_label2_mapping,label2_to_label1_mapping],[label2_to_label3_mapping,label3_to_label2_mapping]]# ?

        self.ratio = ratio
        self.seed = seed
        self.shot = shot
        self.dataset = rcv1_sub_dataset(self.shot, self.seed, self.ratio, ratio_flag=ratio_flag)
        print("length dataset['train']:", len(self.dataset['train']))
        self.train_data = self.get_dataset("train")

        self.dev_data = self.get_dataset("dev")
        self.test_data = self.get_dataset("test")
        self.train_example = self.convert_data_to_examples(self.train_data)
        self.dev_example = self.convert_data_to_examples(self.dev_data)
        self.test_example = self.convert_data_to_examples(self.test_data)
        self.train_inputs = [i[0] for i in self.train_data]
        self.dev_inputs = [i[0] for i in self.dev_data]
        self.test_inputs = [i[0] for i in self.test_data]

        self.size = len(self.train_example) + len(self.test_example)

    def get_tree_info(self):
        flat_slot2value = torch.load(os.path.join(self.data_path, 'slot.pt'))

        value2slot = {}
        num_class = 0
        for s in flat_slot2value:
            for v in flat_slot2value[s]:
                value2slot[v] = s
                if num_class < v:
                    num_class = v
        num_class += 1
        for i in range(num_class):
            if i not in value2slot:
                value2slot[i] = -1

        def get_depth(x):
            depth = 0
            while value2slot[x] != -1:
                depth += 1
                x = value2slot[x]
            return depth

        depth_dict = {i: get_depth(i) for i in range(num_class)}
        max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
        depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}
        return flat_slot2value, value2slot, depth2label
            
    def get_dataset(self, type="train"):
        data = []
        cur_dataset = self.dataset[type]
        length = len(cur_dataset)
        for i in tqdm(range(length)):
            text_a = cur_dataset[i][0]
            label = cur_dataset[i][1]
            data.append([text_a, label])
        return data

    def convert_data_to_examples(self, data):
        examples = []
        for idx, sub_data in enumerate(data):
            examples.append(InputExample(guid=str(idx), text_a=sub_data[0], label=sub_data[1]))

        return examples

class DBPProcessor:

    def __init__(self, ratio=-1, seed=171, shot=-1, ratio_flag=0):
        super().__init__()
        self.name = 'DBpeida'
        label0_list, label1_list, label2_list, label0_label2id, label1_label2id, label2_label2id, label0_to_label1_mapping, label1_to_label2_mapping, label1_to_label0_mapping, label2_to_label1_mapping = get_mapping()
        self.labels = label2_list
        self.coarse_labels = label0_list
        self.description=os.path.join(base_path,"dataset","DBpedia","label_description.json")
        self.all_labels = label0_list + label1_list + label2_list
        self.label_list = [label0_list, label1_list, label2_list]
        self.label0_to_label1_mapping = label0_to_label1_mapping
        self.label1_to_label2_mapping = label1_to_label2_mapping
        self.label1_to_label0_mapping = label1_to_label0_mapping
        self.label2_to_label1_mapping = label2_to_label1_mapping

        self.data_path = os.path.join(base_path, "dataset", "DBpedia")
        self.flat_slot2value, self.value2slot, self.depth2label = self.get_tree_info()
        self.hier_mapping = [[label0_to_label1_mapping, label1_to_label0_mapping], [label1_to_label2_mapping, label2_to_label1_mapping]]

        self.ratio = ratio
        self.seed = seed
        self.shot = shot
        self.dataset = sub_dataset(self.shot, self.seed, self.ratio, ratio_flag=ratio_flag)
        print("length dataset['train']:", len(self.dataset['train']))

        self.train_data = self.get_dataset("train")

        self.dev_data = self.get_dataset("val")
        self.test_data = self.get_dataset("test")
        self.train_example = self.convert_data_to_examples(self.train_data)
        self.dev_example = self.convert_data_to_examples(self.dev_data)
        self.test_example = self.convert_data_to_examples(self.test_data)

        self.train_inputs = [i[0] for i in self.train_data]
        self.dev_inputs = [i[0] for i in self.dev_data]
        self.test_inputs = [i[0] for i in self.test_data]

        self.size = len(self.train_example) + len(self.test_example)

    def get_tree_info(self):
        flat_slot2value = torch.load(os.path.join(self.data_path, 'slot.pt'))
        value2slot = {}
        num_class = 0
        for s in flat_slot2value:
            for v in flat_slot2value[s]:
                value2slot[v] = s
                if num_class < v:
                    num_class = v
        num_class += 1
        for i in range(num_class):
            if i not in value2slot:
                value2slot[i] = -1

        def get_depth(x):
            depth = 0
            while value2slot[x] != -1:
                depth += 1
                x = value2slot[x]
            return depth

        depth_dict = {i: get_depth(i) for i in range(num_class)}
        max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
        depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}
        return flat_slot2value, value2slot, depth2label

    def get_dataset(self, type="train"):
        data = []
        cur_dataset = self.dataset[type]
        length = len(cur_dataset)
        for i in tqdm(range(length)):
            text_a = cur_dataset[i][0]
            label = cur_dataset[i][1]
            data.append([text_a, label])
        return data

    def convert_data_to_examples(self, data):
        examples = []
        for idx, sub_data in enumerate(data):
            examples.append(InputExample(guid=str(idx), text_a=sub_data[0], label=sub_data[1]))

        return examples

PROCESSOR = {
    "wos": WOSProcessor,
    "WebOfScience": WOSProcessor,
    "kgwos":KGWOSProcessor,
    "rcv1": RCV1Processor,
    "dbpedia": DBPProcessor
}

