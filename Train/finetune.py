import sys
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from transformers import XLMRobertaConfig

sys.path.append("../")
import os

if "p" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["p"]

import warnings

warnings.filterwarnings("ignore")
import fitlog
from fastNLP import (
    AccuracyMetric,
    BucketSampler,
    ClassifyFPreRecMetric,
    ConstantTokenNumSampler,
    CrossEntropyLoss,
    DataSetIter,
    FitlogCallback,
    RandomSampler,
    SequentialSampler,
    SortedSampler,
    Trainer,
    WarmupCallback,
    cache_results,
)
from fastNLP.embeddings import BertWordPieceEncoder, RobertaWordPieceEncoder
from fitlog import _committer
from torch import optim, nn
from transformers import XLMRobertaModel, XLNetModel

from pipe import ResPipe

# fitlog.debug()
os.makedirs("./FT_logs", exist_ok=True)
fitlog.set_log_dir("FT_logs")


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="Laptop",
    choices=[
        "Restaurants",
        "Laptop",
        "Tweets",
        "fr",
        "sp",
        "dutch",
    ],
)
parser.add_argument(
    "--data_dir",
    type=str,
    help="the dataset dir name, which can be concat with the dataset arguement",
)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument(
    "--model_name",
    type=str,
    default="bert-en-base-uncased",
    choices=[
        "bert-en-base-uncased",
        "roberta-en",
        "roberta-en-large",
        "xlmroberta-xlm-roberta-base",
        "bert-multi-base-cased",
    ],
)
parser.add_argument("--save_embed", default=1, type=int)
parser.add_argument("--batch_size", default=32, type=int)


args = parser.parse_args()

fitlog.add_hyper_in_file(__file__)
fitlog.add_hyper(args)


print(args)
#######hyper
n_epochs = 40
pool = "max"
smooth_eps = 0.0
dropout = 0.5
#######hyper


model_type = args.model_name.split("-")[0]
if model_type == "bert":
    mask = "[UNK]"
elif model_type == "roberta":
    mask = "<mask>"
elif model_type == "xlnet":
    mask = "<mask>"
elif model_type == "xlmroberta":
    mask = "<mask>"


@cache_results(
    "./caches/data_{}_{}_{}.pkl".format(args.dataset, mask, args.model_name),
    _refresh=False,
)
def get_data():
    data_bundle = ResPipe(model_name=args.model_name, mask=mask).process_from_file(
        os.path.join(args.data_dir, args.dataset)
    )
    print("pppppppppp",data_bundle.get_dataset("train"))
    return data_bundle


data_bundle = get_data()

print(data_bundle)

if args.model_name.split("-")[0] in ("bert", "roberta", "xlnet", "xlmroberta"):
    model_type, args.model_name = (
        args.model_name[: args.model_name.index("-")],
        args.model_name[args.model_name.index("-") + 1 :],
    )

if model_type == "roberta":
    embed = RobertaWordPieceEncoder(
        model_dir_or_name=args.model_name, requires_grad=True, num_aspect=1
    )
elif model_type == "bert":
    embed = BertWordPieceEncoder(model_dir_or_name=args.model_name, requires_grad=True)
elif model_type == "xlnet":
    embed = XLNetModel.from_pretrained(pretrained_model_name_or_path=args.model_name)
elif model_type == "xlmroberta":
    embed = XLMRobertaModel.from_pretrained(
        pretrained_model_name_or_path=args.model_name
    )
    embed.add_adapter("restaurant")
    embed.train_adapter("restaurant")
    embed.set_active_adapters("restaurant")

    training_args = TrainingArguments(
        learning_rate=2e-5,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        # logging_steps=200,
        output_dir="./training_output",
        # overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
    )

    train_data = DataSetIter(
        data_bundle.get_dataset("train"),
        num_workers=2,
        batch_sampler=ConstantTokenNumSampler(
            data_bundle.get_dataset("train").get_field("seq_len").content,
            max_token=2000,
            num_bucket=10,
        ),
    )

    test_data = DataSetIter(
        data_bundle.get_dataset("test"),
        num_workers=2,
        batch_sampler=ConstantTokenNumSampler(
            data_bundle.get_dataset("test").get_field("seq_len").content,
            max_token=2000,
            num_bucket=10,
        ),
    )

    adaptertrainer = AdapterTrainer(
        model=embed,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data
    )

    adaptertrainer.train()

class AspectModel(nn.Module):
    def __init__(self, embed, dropout, num_classes, pool="max"):
        super().__init__()
        assert pool in ("max", "mean")
        self.embed = embed
        self.embed_dropout = nn.Dropout(dropout)
        if hasattr(embed, "embedding_dim"):
            embed_size = embed.embedding_dim
        else:
            embed_size = embed.config.hidden_size
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_size, num_classes),
        )
        self.pool = pool

    def forward(self, tokens, aspect_mask):
        """

        :param tokens:
        :param aspect_mask: bsz x max_len, 1 for aspect
        :return:
        """
        if isinstance(self.embed, BertWordPieceEncoder):
            tokens = self.embed(tokens, None)  # bsz x max_len x hidden_size
        else:
            tokens = self.embed(
                tokens, token_type_ids=None
            )  # bsz x max_len x hidden_size

        if isinstance(tokens, tuple):
            tokens = tokens[0]

        tokens = self.embed_dropout(tokens)

        aspect_mask = aspect_mask.eq(1)
        if self.pool == "mean":
            tokens = tokens.masked_fill(aspect_mask.unsqueeze(-1).eq(0), 0)
            tokens = tokens.sum(dim=1)
            preds = tokens / aspect_mask.sum(dim=1, keepdims=True).float()
        elif self.pool == "max":
            aspect_mask = aspect_mask.unsqueeze(-1).eq(0)  # bsz x max_len x 1
            tokens = tokens.masked_fill(aspect_mask, -10000.0)
            preds, _ = tokens.max(dim=1)
        preds = self.ffn(preds)
        return {"pred": preds}


'''model = AspectModel(
    embed,
    dropout=dropout,
    num_classes=len(data_bundle.get_vocab("target")) - 1,
    pool=pool,
)'''
from transformers import RobertaConfig, RobertaModelWithHeads

config = RobertaConfig.from_pretrained(
    "roberta-base"
)
model = RobertaModelWithHeads.from_pretrained(
    "roberta-base",
    config=config,
)
# Add a new adapter
model.add_adapter("restaurant")
# Add a matching classification head
model.add_classification_head(
    "restaurant",
    num_labels=3,
    id2label={ 0: "neutral", 1: "positive", 2: "negative"}
  )
# Activate the adapter
model.train_adapter("restaurant")
model.set_active_adapters("restaurant")

import numpy as np
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=200,
    output_dir="./training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)

def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}
print("qqqqqqqqqqqqqqqqqqqqq",data_bundle.get_dataset("train"))
print("qqqqqqqqqqqqqqqqqqqqq",type(data_bundle.get_dataset("train")))
print("xxx",data_bundle.get_dataset("train")[0]['tokens'])
print("xxx",data_bundle.get_dataset("train")[0]['target'])
print("xxx",data_bundle.get_dataset("train")[0]['aspect_mask'])
print("xxx",data_bundle.get_dataset("train")[0]['raw_words'])
print("xxx",data_bundle.get_dataset("train")[0]['labels'])
trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=data_bundle.get_dataset("train"),
    eval_dataset=data_bundle.get_dataset("test"),
    compute_metrics=compute_accuracy,
)

trainer.train()
trainer.evaluate()

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 1e-2,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]
optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

callbacks = []
callbacks.append(WarmupCallback(0.01, "linear"))
callbacks.append(
    FitlogCallback(
        # data_bundle.get_dataset('train')
    )
)

import torch
import torch.nn.functional as F
from fastNLP import LossBase


class SmoothLoss(LossBase):
    def __init__(self, smooth_eps=0):
        super().__init__()
        self.smooth_eps = smooth_eps

    def get_loss(self, pred, target):
        """

        :param pred: bsz x 3
        :param target: bsz,
        :return:
        """
        n_class = pred.size(1)
        smooth_pos = target.eq(n_class)
        target = target.masked_fill(smooth_pos, 0)
        target_matrix = torch.full_like(
            pred, fill_value=self.smooth_eps / (n_class - 1)
        )
        target_matrix = target_matrix.scatter(
            dim=1, index=target.unsqueeze(1), value=1 - self.smooth_eps
        )
        target_matrix = target_matrix.masked_fill(
            smooth_pos.unsqueeze(1), 1.0 / n_class
        )

        pred = F.log_softmax(pred, dim=-1)
        loss = -(pred * target_matrix).sum(dim=-1).mean()
        return loss



tr_data = DataSetIter(
    data_bundle.get_dataset("train"),
    num_workers=2,
    batch_sampler=ConstantTokenNumSampler(
        data_bundle.get_dataset("train").get_field("seq_len").content,
        max_token=2000,
        num_bucket=10,
    ),
)


'''trainer = Trainer(
    tr_data,
    model,
    optimizer=optimizer,
    loss=SmoothLoss(smooth_eps),
    batch_size=args.batch_size,
    sampler=BucketSampler(),
    drop_last=False,
    update_every=32 // args.batch_size,
    num_workers=2,
    n_epochs=n_epochs,
    print_every=5,
    dev_data=data_bundle.get_dataset("test"),
    metrics=[AccuracyMetric(), ClassifyFPreRecMetric(f_type="macro")],
    metric_key=None,
    validate_every=-1,
    save_path=None,
    use_tqdm=False,
    device=0,
    callbacks=callbacks,
    check_code_level=0,
    test_sampler=SortedSampler(),
    test_use_tqdm=False,
)'''

#trainer.train(load_best_model=False)


fitlog.add_other(trainer.start_time, name="start_time")
os.makedirs("./save_models", exist_ok=True)
folder = "./save_models/{}-{}-{}".format(model_type, args.dataset, "FT")
count = 0
for fn in os.listdir("save_models"):
    if fn.startswith(folder.split("/")[-1]):
        count += 1
folder = folder + str(count)
fitlog.add_other(count, name="count")
if args.save_embed and not os.path.exists(folder):
    if not isinstance(embed, XLMRobertaModel):
        embed.save(folder)
    else:
        os.makedirs(folder, exist_ok=True)
        os.makedirs("{}/{}".format(folder, model_type), exist_ok=True)
        embed.save_pretrained("{}/{}".format(folder, model_type))
