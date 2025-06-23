import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
from requests.exceptions import HTTPError, ConnectionError, Timeout

import datasets as hf_datasets
import evaluate as hf_evaluate
import torch

from moe_peft.common import InputData, Prompt

def load_dataset_with_retry(dataset_name, max_retries=5, retry_delay=5, *args, **kwargs):
    attempt = 0
    
    while attempt < max_retries:
        try:
            dataset = hf_datasets.load_dataset(dataset_name, *args, **kwargs)
            return dataset
        except HTTPError as e:
            # HTTP status codes that are typically resolved by retrying
            retryable_status_codes = [429, 500, 502, 503, 504]
            if e.response.status_code in retryable_status_codes:
                attempt += 1
                retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    # If server sent a 'Retry-After' header, use its value
                    retry_delay = int(retry_after)
                print(f"Attempt {attempt}/{max_retries} failed with {e.response.status_code} error. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise
        except (ConnectionError, Timeout) as e:
            attempt += 1
            print(f"Attempt {attempt}/{max_retries} failed due to network issue ({e}). Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    raise RuntimeError(f"Failed to load dataset {dataset_name} after {max_retries} retries.")


class BasicMetric:
    def __init__(self) -> None:
        pass

    def add_batch(self, predictions: torch.Tensor, references: torch.Tensor):
        pass

    def compute(self) -> Dict[str, Any]:
        pass


class AutoMetric(BasicMetric):
    def __init__(self, task_name: str) -> None:
        super().__init__()
        path_prefix = os.getenv("MOE_PEFT_METRIC_PATH")
        if path_prefix is None:
            path_prefix = ""
        elif not path_prefix.endswith(os.sep):
            path_prefix += os.sep

        if ":" in task_name:
            split = task_name.split(":")
            self.metric_ = hf_evaluate.load(path_prefix + split[0], split[1])
        else:
            self.metric_ = hf_evaluate.load(path_prefix + task_name)

    def add_batch(self, predictions: torch.Tensor, references: torch.Tensor):
        self.metric_.add_batch(predictions=predictions, references=references)

    def compute(self) -> Dict[str, Any]:
        return self.metric_.compute()


class BasicTask:
    def __init__(self) -> None:
        pass

    @property
    def peft_task_type(self) -> str:
        pass

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        pass

    def loading_metric(self) -> BasicMetric:
        pass

    def init_kwargs(self) -> Dict:
        return {}


# Casual Fine-tuning Tasks
# Instant-Created Class
class CasualTask(BasicTask):
    @property
    def peft_task_type(self) -> str:
        return "CAUSAL_LM"

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        assert path is not None, "Casual supervised fine-tuning requires data path."
        assert is_train, "Casual supervised fine-tuning task only supports training."
        # Loading dataset
        if path.endswith(".json") or path.endswith(".jsonl"):
            data = hf_datasets.load_dataset("json", data_files=path)
        elif ":" in path:
            split = path.split(":")
            data = hf_datasets.load_dataset(split[0], split[1])
        else:
            data = hf_datasets.load_dataset(path)
        ret: List[InputData] = []
        for data_point in data["train"]:
            ret.append(
                InputData(
                    inputs=Prompt(
                        instruction=data_point["instruction"],
                        input=data_point.get("input", None),
                        label=data_point.get("output", None),
                    )
                )
            )

        return ret


# Sequence Classification
class SequenceClassificationTask(BasicTask):
    def __init__(
        self,
        task_name: str,
        task_type: str,
        label_dtype: torch.dtype,
        num_labels: int,
        dataload_function: Callable,
        # Setting to `None` corresponds to the task name.
        metric_name: Optional[str] = None,
        # The default values are "train" and "validation".
        subset_map: Optional[Tuple[str, str]] = ("train", "validation"),
        task_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.task_name_ = task_name
        self.task_type_ = task_type
        self.label_dtype_ = label_dtype
        self.num_labels_ = num_labels
        self.dataload_function_ = dataload_function
        if metric_name is None:
            self.metric_name_ = task_name
        else:
            self.metric_name_ = metric_name
        self.subset_map_ = subset_map
        self.task_id_ = task_id

    @property
    def peft_task_type(self) -> str:
        return "SEQ_CLS"

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        if ":" in self.task_name_:
            split = self.task_name_.split(":")
            # data = hf_datasets.load_dataset(
            #     split[0] if path is None else path, split[1]
            # )
            data = load_dataset_with_retry(
                split[0] if path is None else path, 
                name=split[1],
            )
        else:
            # data = hf_datasets.load_dataset(self.task_name_ if path is None else path)
            data = load_dataset_with_retry(
                self.task_name_ if path is None else path
            )
        data = data[self.subset_map_[0] if is_train else self.subset_map_[1]]
        logging.info(f"Preparing data for {self.task_name_.upper()}")
        ret: List[InputData] = []
        for data_point in data:
            inputs, labels = self.dataload_function_(data_point)
            assert isinstance(labels, List)
            ret.append(InputData(inputs=inputs, labels=labels, task_id=self.task_id_))

        return ret

    def loading_metric(self) -> BasicMetric:
        return AutoMetric(self.metric_name_)

    def init_kwargs(self) -> Dict:
        return {
            "task_type": self.task_type_,
            "num_labels": self.num_labels_,
            "label_dtype": self.label_dtype_,
        }


# Common Sense
class CommonSenseTask(BasicTask):
    def __init__(self) -> None:
        super().__init__()
        self.task_type_ = "common_sense"
        self.label_dtype_ = None

    @property
    def peft_task_type(self) -> str:
        return "QUESTION_ANS"

    def label_list(self) -> List[str]:
        pass


task_dict = {}


# Multi-Task (Only for train)
class MultiTask(BasicTask):
    def __init__(self, task_names: str) -> None:
        super().__init__()
        self.task_type_ = "multi_task"
        self.label_dtype_ = None
        self.task_list_: List[BasicTask] = []
        task_names = task_names.split(";")
        for name in task_names:
            self.task_list_.append(task_dict[name])

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        logging.info(f"Preparing data for {len(self.task_list_)} tasks")
        path_list = None if path is None else path.split(";")
        data: List[InputData] = []
        assert is_train
        for idx, task in enumerate(self.task_list_):
            path: str = "" if path_list is None else path_list[idx].strip()
            data.extend(task.loading_data(is_train, None if len(path) == 0 else path))
        return data
