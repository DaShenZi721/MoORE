import torch

from moe_peft.common import TASK_IDS
from .common import SequenceClassificationTask


def update_task_dict(task_dict):
    task_dict.update(
        {
            "glue:cola": SequenceClassificationTask(
                task_name="glue:cola",
                task_type="single_label_classification",
                num_labels=2,
                label_dtype=torch.long,
                # dataload_function=lambda data_point: (
                #     [data_point["sentence"]],
                #     [int(data_point["label"])],
                # ),
                dataload_function=lambda data_point: (
                    f'Is the following sentence ”{data_point["sentence"]}” grammatically acceptable?\nAnswer:',
                    [int(data_point["label"])],
                ),
                task_id=TASK_IDS["glue:cola"],
            ),
            "glue:mnli": SequenceClassificationTask(
                task_name="glue:mnli",
                task_type="single_label_classification",
                num_labels=3,
                label_dtype=torch.long,
                # dataload_function=lambda data_point: (
                #     [data_point["premise"], data_point["hypothesis"]],
                #     [int(data_point["label"])],
                # ),
                dataload_function=lambda data_point: (
                    f'Dose the statement ”{data_point["premise"]}” imply that ”{data_point["hypothesis"]}”?\nAnswer:',
                    [int(data_point["label"])],
                ),
                subset_map=("train", "validation_matched"),
                task_id=TASK_IDS["glue:mnli"],
            ),
            "glue:mrpc": SequenceClassificationTask(
                task_name="glue:mrpc",
                task_type="single_label_classification",
                num_labels=2,
                label_dtype=torch.long,
                # dataload_function=lambda data_point: (
                #     [data_point["sentence1"], data_point["sentence2"]],
                #     [int(data_point["label"])],
                # ),
                dataload_function=lambda data_point: (
                    f'Dose the following sentence ”{data_point["sentence1"]}” convey the equivalent meaning as ”{data_point["sentence2"]}”?\nAnswer:',
                    [int(data_point["label"])],
                ),
                task_id=TASK_IDS["glue:mrpc"],
            ),
            "glue:qnli": SequenceClassificationTask(
                task_name="glue:qnli",
                task_type="single_label_classification",
                num_labels=2,
                label_dtype=torch.long,
                # dataload_function=lambda data_point: (
                #     [data_point["question"], data_point["sentence"]],
                #     [int(data_point["label"])],
                # ),
                dataload_function=lambda data_point: (
                    f'Based on the statement: ”{data_point["question"]}” dose the following sentence ”{data_point["sentence"]}” have a definitive answer?\nAnswer:',
                    [int(data_point["label"])],
                ),
                task_id=TASK_IDS["glue:qnli"],
            ),
            "glue:qqp": SequenceClassificationTask(
                task_name="glue:qqp",
                task_type="single_label_classification",
                num_labels=2,
                label_dtype=torch.long,
                # dataload_function=lambda data_point: (
                #     [data_point["question1"], data_point["question2"]],
                #     [int(data_point["label"])],
                # ),
                dataload_function=lambda data_point: (
                    f'Is the following question ”{data_point["question1"]}” essentially asking the same thing as ”{data_point["question2"]}”?\nAnswer:',
                    [int(data_point["label"])],
                ),
                task_id=TASK_IDS["glue:qqp"],
            ),
            "glue:rte": SequenceClassificationTask(
                task_name="glue:rte",
                task_type="single_label_classification",
                num_labels=2,
                label_dtype=torch.long,
                # dataload_function=lambda data_point: (
                #     [data_point["sentence1"], data_point["sentence2"]],
                #     [int(data_point["label"])],
                # ),
                dataload_function=lambda data_point: (
                    f'Dose the text ”{data_point["sentence1"]}” entail the statement ”{data_point["sentence2"]}”?\nAnswer:',
                    [int(data_point["label"])],
                ),
                task_id=TASK_IDS["glue:rte"],
            ),
            "glue:sst2": SequenceClassificationTask(
                task_name="glue:sst2",
                task_type="single_label_classification",
                num_labels=2,
                label_dtype=torch.long,
                # dataload_function=lambda data_point: (
                #     [data_point["sentence"]],
                #     [int(data_point["label"])],
                # ),
                dataload_function=lambda data_point: (
                    f'Is the following sentence ”{data_point["sentence"]}” sentimently positive?\nAnswer:',
                    [int(data_point["label"])],
                ),
                task_id=TASK_IDS["glue:sst2"],
            ),
            "glue:wnli": SequenceClassificationTask(
                task_name="glue:wnli",
                task_type="single_label_classification",
                num_labels=2,
                label_dtype=torch.long,
                dataload_function=lambda data_point: (
                    [data_point["sentence1"] + " </s> " + data_point["sentence2"]],
                    [int(data_point["label"])],
                ),
                task_id=TASK_IDS["glue:wnli"],
            ),
        }
    )
