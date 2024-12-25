from transformers import TrainingArguments, HfArgumentParser
from dataclasses import dataclass, field
import typing


def StrBoolTupleType(arg_str: str) -> typing.Tuple[str, bool]:
    if "," in arg_str:
        s, b = arg_str.split(",")
        return str(s), (b.lower() in ("true", "1"))
    else:
        return arg_str, False


@dataclass
class ModelArguments:
    model_1_name: typing.Optional[str] = field(
        default=None,
        metadata={"help": "model URI or path to finetune."}
    )
    model_2_name: typing.Optional[str] = field(
        default=None,
        metadata={"help": "model URI or path to finetune."}
    )
    use_lk: bool = field(
        default=True,
        metadata={
            "help": "Whether to use the Liger kernel.",
            "aliases": ["--k"]
        }
    )
    end_step: int = 500


@dataclass
class DatasetArguments:
    dataset_max_seq_length: typing.Optional[None] = 2048
    dataset_test_size: float = 0.002
    dataset_shuffle: bool = False
    dataset_shuffle_seed: int = 42
    


@dataclass
class EvalArguments:
    grad_var_stats: bool = True
    binary_grad_similarity_stats: bool = False
    full_grad_similarity_stats: bool = False  # expensive

    harness_benchmarks: typing.List[typing.Dict] = field(
        default_factory= lambda: ["arc_challenge"],
        # official model release recommendation:
        # include lambda: ["wikitext", "boolq", "hellaswag", "glue", "ai2_arc", "mmlu", "math"]
        metadata={"help": "Benchmarks to compare student and teacher models at end of training."}
    )
    harness_benchmark_limit: int = field(
        default=5000,
        # official model release recommendation: set to None for official releases to measure all data points
        metadata={"help": "Limit the number of examples per task (only use this for testing), If <1, limit is %."}
    )
    harness_benchmark_bootstrap_iters: int = field(
        default=0,
        # official model release recommendation: set to None for official releases to measure error
        metadata={"help": "Number iter for bootstrap stats for stderr. Set to 0 to skip stderr calc. "}
    )

@dataclass
class TrainingArguments(TrainingArguments):
    # optimize convergence to final model
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "linear"
    # lr_scheduler_kwargs: dict = field(default_factory=lambda: {"lr_end": 1e-5})
    num_train_epochs: float = 1.0
    optim: str = "paged_adamw_8bit"
    # optim_args: dict = {"betas": (0.9, 0.999), "eps": 1e-08}
    max_steps: int = 5000
    # max_grad_norm: float = 0.5
    distil: bool = True
    distmodel: str = "semran1/llama3-3b-1210" # change to whichever model you are using

    # DDP thing
    ddp_find_unused_parameters: bool = False #True
    # larger batches appear to train better?
    per_device_train_batch_size: int = 3
    gradient_accumulation_steps: int = 5

    # optimize performance and memory
    per_device_eval_batch_size: int = 8  # TODO: auto-find?
    gradient_checkpointing: bool = True
    bf16: bool = True

    # TODO: enable torch compile when this incompatibility with use_liger_kernel is fixed
    # -----------------------
    # /opt/conda/lib/python3.10/site-packages/torch/autograd/graph.py:825:
    # UserWarning: cuDNN SDPA backward got grad_output.strides() != output.strides(),
    # attempting to materialize a grad_output with matching strides...
    # (Triggered internally at ../aten/src/ATen/native/cudnn/MHA.cpp:674.)
    # torch_compile: bool = True  # TODO: Field

    # Fixes
    gradient_checkpointing_kwargs = {"use_reentrant": False}

    # logging / evaluation
    logging_steps: int = 1
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit = 4
    eval_strategy: str = "steps"
    eval_steps: int = 500000000
    eval_on_start: bool = False
    eval_on_end: bool = True
    report_to: str = "wandb"
    run_name: str = "merge-run"
    resume: bool = False


parser = HfArgumentParser((
    TrainingArguments,
    ModelArguments,
    DatasetArguments,
    EvalArguments
))


def get_args():
    return parser.parse_args_into_dataclasses()
