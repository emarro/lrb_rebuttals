import argparse
import datasets
import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk 
from datasets.distributed import split_dataset_by_node
from accelerate import Accelerator
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)


def load_model_and_tokenizer(model_name):
    # model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, revision="main"
    )
    model_config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, revision="main"
    )

    model = AutoModelForMaskedLM.from_pretrained(
        model_name, config=model_config, trust_remote_code=True, revision="main"
    )
    model = model
    return model, tokenizer

def chr_to_int(chromosome):
    chr_map = {str(x): x for x in range(23)}
    chr_map['X'] = 24
    chr_map['Y'] = 25
    return chr_map[chromosome]


# ## Load dataset
# Load the test split of the LRB dataset
def load_lrb_dataset(sequence_length, task_name):
    sequence_length = sequence_length  # 131k

    # One of:
    # ["variant_effect_causal_eqtl","variant_effect_pathogenic_clinvar",
    # "variant_effect_pathogenic_omim","cage_prediction", "bulk_rna_expression",
    # "chromatin_features_histone_marks","chromatin_features_dna_accessibility",
    # "regulatory_element_promoter","regulatory_element_enhancer"]

    task_name = task_name
    subset = False
    if task_name == "variant_effect_pathogenic_omim":
        subset = True

    dataset = load_dataset(
        "InstaDeepAI/genomics-long-range-benchmark",
        task_name=task_name,
        sequence_length=sequence_length,
        split="test",
        streaming=False,
        subset=subset,
        num_proc=4
    )
    #dataset = dataset.with_format("torch")
    return dataset


# %% [markdown]
# ## Forward Pass Wrappers
# Wrappers to unpack the data and calculate:
# $$\text{Variant Effect Score} = \log{\frac{P_\text{ref}}{P_{\text{alt}}}} $$
# The method to obtain the probabilities differ slightly between autoregressive models and Bert like (MLM) models so we define a function for each type of model.


# %%
@torch.no_grad()
def calc_ar_zeroshot(model, tokenizer, batched_data):
    """
    Calcuclate zero-shot pred for a single variant using a autoregressive model
    """
    raise NotImplementedError()
    ref = tokenizer(batched_data["ref_forward_sequence"])  # [3, batch, seq_len]
    ref_ids = torch.Tensor(ref["input_ids"]).int()  # [batch, seq_len]
    ref_attn_mask = ref["attention_mask"]  # [batch, seq_len]

    alt = tokenizer(batched_data["alt_forward_sequence"])  # [3, batch, seq_len]
    alt_ids = torch.Tensor(alt["input_ids"]).int()  # [batch, seq_len]
    alt_attn_mask = alt["attention_mask"]  # [batch, seq_len]

    variant_pos = torch.Tensor(
        [1024 for _ in range(ref_ids.size(0))]
    ).int()  # batched_data['position'] # [batch]
    ref_token = ref_ids[:, variant_pos]  # [batch]
    alt_token = alt_ids[:, variant_pos]  # [batch]

    probs = model(input_ids=ref_ids)  # [batch, seq_len, vocab_size]
    return probs

    ref_probs = probs[:, variant_pos, ref_token]  # [batch]
    alt_probs = probs[:, variant_pos, alt_token]  # [batch]
    return (ref_probs / alt_probs).log()  # [batch]


# %%
@torch.no_grad()
def calc_mlm_zeroshot(model_name, model, tokenizer, batched_data, batch_size):
    """
    Calcuclate zero-shot pred for a single variant for a bert style model (e.g. caduceus)
    """
    #Get sequence input ids
    ref_ids = batched_data["input_ids"].long()  # [batch, seq_len]
    alt_ids = batched_data["alt_input_ids"].long()  # [batch, seq_len]

    batch_size, seq_len = ref_ids.size()

    # assume the variant of interest is in the middle of the sequence
    mask_location = seq_len // 2
    #print(ref_ids[0, mask_location-2:mask_location+2])
    #print(alt_ids[0, mask_location-2:mask_location+2])

    #find the ref and alt tokens for each batch
    ref_token = torch.clone(ref_ids[:, mask_location]).detach()  # [batch]
    alt_token = torch.clone(alt_ids[:, mask_location]).detach()  # [batch]

    #make sure our mask_location makes sense
    assert (
        ref_ids[0, mask_location].item() != alt_ids[0, mask_location].item()
    ), f"The ref and alternate sequence have the same token at {mask_location}, the masking location is likely incorrect."

    # mask out the position of the variant of interest
    ref_ids[:, mask_location] = tokenizer.mask_token_id
    ref_ids = ref_ids.long()

    alt_ids[:, mask_location] = tokenizer.mask_token_id
    alt_ids = alt_ids.long()
    assert ref_token[0] != alt_token[0], f"Ref token {ref_token[0]} and alt token {alt_token[0]} are the same when they should be different"


    # Forward pass
    if 'ph' in model_name: #need to avg hidden_states:
        outs = model(input_ids=ref_ids)
        print(outs)
        fwd_hidden_state=outs['hidden_states']
        print(fwd_hidden_state.shape)
        #TODO: pass in RC, take those hidden, avg, then call lm_head directly
        assert False
    probs = model(input_ids=ref_ids)["logits"].softmax(
        dim=-1
    )  # [batch, seq_len, vocab_size]
    #print(probs)

    # select P_ref and P_alt
    ref_probs = []
    alt_probs = []
    for b_idx in range(batch_size):
        ref_prob = probs[b_idx, mask_location, ref_token[b_idx]]  # [1]
        ref_probs.append(ref_prob)
        alt_prob = probs[b_idx, mask_location, alt_token[b_idx]]  # [1]
        alt_probs.append(alt_prob)
    ref_probs = torch.stack(ref_probs, dim=0)  # [batch]
    alt_probs = torch.stack(alt_probs, dim=0)  # [batch]
    #print(ref_token)
    #print(alt_token)
    #print(ref_probs)
    #print(alt_probs)

    labels = batched_data["label"]  # [batch]

    #return the variant effect score
    score = (
        ref_probs / alt_probs
    ).log()
    #print(score)
    return score, labels  # ([batch], [batch]) == (y_score, y_true)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Run zero shot eval on a LRB task for a Bert style model"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The HF model name to load. e.g.: (kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16), (kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16), (kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3), (kuleshov-group/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3) ",
    )
    zero_shot_tasks = [
        "variant_effect_causal_eqtl",
        "variant_effect_pathogenic_clinvar",
        "variant_effect_pathogenic_omim",
    ]
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help=f"The zero shot task of LRB to use, should be one of: {zero_shot_tasks}",
    )

    parser.add_argument(
        "--sequence_length",
        type=int,
        required=True,
        help=f"The sequence length to use to request for the task from LRB (e.g. {2**17} (131k), {2**10} (1k))",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The size of the batch for each forward pass",
    )
    
    parser.add_argument(
        "--shards",
        type=int,
        default=None,
        help="The number of shards to split the dataset up into",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="The rank (shard_idx) of this process",
    )


    args = parser.parse_args()

    model_name = args.model_name
    model, tokenizer = load_model_and_tokenizer(model_name=model_name)

    zero_shot_tasks = [
        "variant_effect_causal_eqtl",
        "variant_effect_pathogenic_clinvar",
        "variant_effect_pathogenic_omim",
    ]

    task_name = args.task_name
    sequence_length = args.sequence_length
    datasets.enable_caching()
    cache_ds = task_name+"_"+str(sequence_length)+".arrow"
    shards = args.shards
    rank = args.rank
    if shards is not None and rank is not None:
        cache_ds = task_name+"_"+str(sequence_length)+"_" +str(rank) + ".arrow"
    accelerator = Accelerator()
    batch_size = args.batch_size 
    if not os.path.exists(cache_ds) and accelerator.is_main_process:
        dataset = load_lrb_dataset(sequence_length=sequence_length, task_name=task_name)
        if shards is not None and rank is not None:
            dataset = dataset.shard(num_shards=shards, index=rank)
            task_name = task_name+"_"+str(rank)

        #dataset = split_dataset_by_node(dataset=dataset, rank=accelerator.local_process_index, world_size=accelerator.num_processes)
        print(dataset)
        dataset = dataset.map(lambda x: tokenizer(x["ref_forward_sequence"]), batched=True, batch_size=256)
        dataset = dataset.map(
            lambda x: {"chromosome": list(map(chr_to_int, x["chromosome"]))}, batched=True, batch_size=256
        )
        dataset = dataset.map(
            lambda x: {
                "alt_" + k: v for k, v in (tokenizer(x["alt_forward_sequence"])).items()
            },
            batched=True, 
            batch_size=512
        )
        example = next(iter(dataset))
        rm_cols = [k for k, v in example.items() if type(v) is str]
        dataset = dataset.remove_columns(rm_cols)
        dataset = dataset.with_format(type="torch")
        dataset.save_to_disk(cache_ds)
    else:
        dataset = load_from_disk(cache_ds)

    dl = DataLoader(dataset, batch_size=batch_size)

    

    (model, dl) = accelerator.prepare(model, dl)

    out_agg, label_agg = [], []
    for data in tqdm(dl):
        with accelerator.no_sync(model):
            with accelerator.autocast():
                out, label = calc_mlm_zeroshot(model_name, model, tokenizer, data, batch_size=batch_size)
        out_agg.append(out)
        label_agg.append(label)
    out_agg = accelerator.gather_for_metrics(out_agg)
    label_agg = accelerator.gather_for_metrics(label_agg)
    print(len(out_agg))
    print(out_agg[0].shape)
    print(len(label_agg))


    if accelerator.is_main_process:
        print(len(out_agg))
        print(out_agg[0].shape)
        print(len(label_agg))

        # %% [markdown]
        # ## Save Outputs
        # Save the outputs of all the evaluated metrics and compute some performance metrics

        # %%
        var_scores = torch.cat(out_agg, dim=0).cpu().detach()
        print(f"Var scores shape {var_scores.shape}")
        scores_path = model_name.split("/")[1] + "_" + task_name + "_zero_shot_scores.pt"
        torch.save(var_scores, scores_path)

        # %%
        var_labels = torch.cat(label_agg, dim=0).cpu().detach()
        print(f"Var labels shape {var_labels.shape}")
        labels_path = model_name.split("/")[1] + "_" + task_name + "_zero_shot_labels.pt"
        torch.save(var_scores, labels_path)

        # %%
        roc_auc = roc_auc_score(y_true=var_labels, y_score=var_scores)
        roc_score_path = (
            model_name.split("/")[1] + "_" + task_name + "_zero_shot_roc_auc.pt"
        )
        torch.save(roc_auc, roc_score_path)
        print(f"Model {model_name} achieved an aucroc of {roc_auc} on task: {task_name}")

        # %%
        pr_auc = average_precision_score(y_true=var_labels, y_score=var_scores)
        pr_score_path = model_name.split("/")[1] + "_" + task_name + "_zero_shot_pr_auc.pt"
        torch.save(pr_auc, pr_score_path)
        print(f"Model {model_name} achieved an aucPR of {pr_auc} on task: {task_name}")
