from dataclasses import asdict
import statistics
import collections
import os
import gc
import shelve

import transformers
from dataclasses import asdict
import statistics
import collections
import os
import gc
import shelve

import transformers
import torch
from huggingface_hub import ModelCard
from args import *
from mergelinear import *
from data import *
from metrics import *
from modelcard import *
from models import *
import objectives
import datasets
from torch.optim.lr_scheduler import LambdaLR
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

def zero_grad(grad):
    return grad.zero_()

class MergeTrainer(transformers.Trainer):
    def __init__(
        self,
        objective,
        model,
        tokenizer,
        evaluators,
        all_args=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, model=model, tokenizer=tokenizer, **kwargs)

        self.all_args = all_args or {}

        self.objective = objective

        self.evaluators = evaluators

        self._prev_grad = None
        self._prev_grad_8bit = None
        self._prev_grad_4bit = None
        self._prev_grad_sign = None
        self._extra_stats = []
        self._rolling_grad_norms = collections.deque(maxlen=16)

    @classmethod
    def from_args(
            cls,
            training_args,
            model_args,
            dataset_args,
            eval_args
    ):
        model_kwargs = {}
        if training_args.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif training_args.fp16:
            model_kwargs["torch_dtype"] = torch.float16

        model, tokenizer = get_model_tokenizer(model_args, **model_kwargs)
        print("got model")
        model.model.embed_tokens.weight.register_hook(zero_grad)
        model.lm_head.weight.register_hook(zero_grad)
        # ensure train data doesn't extend beyond the bounds of the model
        if not dataset_args.dataset_max_seq_length:
            dataset_args.dataset_max_seq_length = tokenizer.model_max_length
        else:
            dataset_args.dataset_max_seq_length = min(dataset_args.dataset_max_seq_length, tokenizer.model_max_length)
        # change this
        evaluators = None # we're just gonna run hellaswag on checkpoints lol!

        print("starting datasets")
        train_dataset = get_dataset(dataset_args, tokenizer)
        test_dataset = get_dataset(dataset_args, tokenizer)

        objective = objectives.Objective(training_args.distil, training_args.distmodel)

        return cls(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            objective=objective,
            evaluators=evaluators,
            all_args=dict(
                model_args=model_args,
                dataset_args=dataset_args,
                eval_args=eval_args,
            )
        )

    @classmethod
    def from_kwargs(cls, **kwargs):
        parsed_args_tuple = parser.parse_dict(
            kwargs,
            allow_extra_keys=False
        )
        return cls.from_args(*parsed_args_tuple)

    def train(self, *args, **kwargs):
        train_output = super().train(*args, **kwargs)
        if self.args.eval_on_end:
            self.evaluate()
        bench_metrics_out = self._maybe_benchmark()
        if bench_metrics_out is None:
            return train_output
        return transformers.trainer_utils.TrainOutput(
            train_output.global_step,
            train_output.training_loss,
            {**train_output.metrics, **bench_metrics_out}
        )

    def compute_loss(self, model, inputs, return_outputs=False, return_stats=False):
        del inputs["labels"]

        loss_dict = self.objective.forward(model, inputs)
        loss = loss_dict.pop("loss")

        stats = {k: float(v) for k, v in loss_dict.items()}

        if return_outputs:
            # TODO: real output, this is nothing of use
            return loss, torch.tensor([1.0])
        elif stats:
            return loss, stats
        else:
            return loss

    def training_step(self, model, inputs) -> torch.Tensor:
        """
        Copy of https://github.com/huggingface/transformers/blob/52a021375/src/transformers/trainer.py#L3394

        With additional behavior:
        - Enable comparative gradient logging
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        if transformers.trainer.is_sagemaker_mp_enabled():
            loss_mb = transformers.trainer.smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss, stats = self.compute_loss(model, inputs, return_stats=True)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if transformers.trainer.is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif transformers.trainer.is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif transformers.trainer.is_torch_musa_available():
                torch.musa.empty_cache()
            elif transformers.trainer.is_torch_npu_available():
                torch.npu.empty_cache()
            elif transformers.trainer.is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [transformers.trainer.OptimizerNames.LOMO, transformers.trainer.OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with transformers.trainer.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        self._extra_stats.append(stats)

        ##############
        # END NEW CODE
        ##############
        self.objective.count = self.state.global_step

        return loss.detach() / self.args.gradient_accumulation_steps

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        """
        Copy of https://github.com/huggingface/transformers/blob/52a021375/src/transformers/trainer.py#L3394

        With additional behavior:
        - Enable gradient variance logging
        - clear self._extra_stats once queued for logging
        """
        
        self._rolling_grad_norms.append(grad_norm.item())
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if transformers.trainer.is_torch_xla_available():
                transformers.trainer.xm.mark_step()

            logs = {}

            ##############
            # NEW CODE
            ##############

            transposed_stats = collections.defaultdict(list)
            [transposed_stats[key].append(d.get(key)) for d in self._extra_stats for key in d]
            for k in transposed_stats:
                if k[0] != "_":
                    logs[k] = sum(transposed_stats[k]) / len(transposed_stats[k])

            if len(self._rolling_grad_norms) == 16 and self.all_args["eval_args"].grad_var_stats:
                logs["grad_norm_var"] = statistics.variance(self._rolling_grad_norms)

            self._extra_stats = []

            ##############
            # END NEW CODE
            ##############

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def evaluate(self, *args, metric_key_prefix="eval", **kwargs):
        self.model.eval()
        gc.collect()
        torch.cuda.empty_cache()
        metrics = {}
        results = (self.bench_single())
        print(results)
        self.model.train()
        gc.collect()
        torch.cuda.empty_cache()
        return metrics

    def create_model_card(self, *args, **kwargs):
        super().create_model_card(*args, **kwargs)

        model_card_filepath = os.path.join(self.args.output_dir, "README.md")
        model_card = ModelCard.load(model_card_filepath)
        model_card = update_model_card(model_card, self) # distily.modelcard.update_model_card(model_card, self)
        model_card.save(model_card_filepath)

    def _maybe_benchmark(self):
        if not self.all_args.get("eval_args") or not self.all_args["eval_args"].harness_benchmarks:
            return

        benchmarks = self.all_args["eval_args"].harness_benchmarks
        limit = self.all_args["eval_args"].harness_benchmark_limit
        bootstrap_iters = self.all_args["eval_args"].harness_benchmark_bootstrap_iters

        self.model.eval()
        self.teacher_model.eval()
        gc.collect()
        torch.cuda.empty_cache()

        with shelve.open(self.benchmarks_shelf) as db:
            if "teacher" not in db:
                # db["teacher"] = distily.metrics.run_benchmarks(
                #     self.teacher_model, self.tokenizer, benchmarks, limit, bootstrap_iters
                # )
                db["teacher"] = run_benchmarks(
                    self.teacher_model, self.tokenizer, benchmarks, limit, bootstrap_iters
                )
            # student_metrics = distily.metrics.run_benchmarks(
            #     self.model, self.tokenizer, benchmarks, limit, bootstrap_iters
            # )
            student_metrics = run_benchmarks(
                self.model, self.tokenizer, benchmarks, limit, bootstrap_iters
            )
            db[self.args.run_name] = student_metrics
            print(student_metrics)

        return student_metrics
    
    def bench_single(self):
        if not self.all_args.get("eval_args") or not self.all_args["eval_args"].harness_benchmarks:
            return

        benchmarks = self.all_args["eval_args"].harness_benchmarks
        limit = self.all_args["eval_args"].harness_benchmark_limit
        bootstrap_iters = self.all_args["eval_args"].harness_benchmark_bootstrap_iters

        self.model.eval()
        gc.collect()
        torch.cuda.empty_cache()

        with shelve.open(self.benchmarks_shelf) as db:
            student_metrics = run_benchmarks(
                self.model, self.tokenizer, benchmarks, limit, bootstrap_iters
            )
            #  db[self.args.run_name] = student_metrics
        self.model.train()
        return student_metrics

    @property
    def benchmarks_shelf(self):
        return os.path.join(self.args.output_dir, "benchmarks.shelve")
