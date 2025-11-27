"""Implement HuggingfaceModel models."""
import copy
import logging
import os
from collections import Counter

import accelerate
import torch
from accelerate import Accelerator

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download


from semantic_uncertainty.uncertainty.models.base_model import BaseModel
from semantic_uncertainty.uncertainty.models.base_model import STOP_SEQUENCES


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None, device="cpu"):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        self.device = device
        if self.match_on == 'tokens':
            # push stop token sequences to the same device as the model/inputs
            self.stops = [
                torch.tensor(self.tokenizer.encode(i)).to(self.device)
                for i in self.stops
            ]
            print(self.stops)


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise
            if match:
                return True
        return False


def remove_split_layer(device_map_in):
    """Modify device maps s.t. individual layers are not spread across devices."""

    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())

    counts = Counter(['.'.join(i.split('.')[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            # Only triggers if we find more than one split layer!
            raise ValueError(
                'More than one split layer.\n'
                f'Currently at layer {layer}.\n'
                f'In map: {device_map_in}\n'
                f'Out map: {device_map}\n')

        logging.info(f'Split layer is {layer}.')

        # remove split for that layer
        for name in list(device_map.keys()):
            if name.startswith(layer):
                print(f'pop {name}')
                device = device_map.pop(name)

        device_map[layer] = device
        found_split = True

    return device_map


class HuggingfaceModel(BaseModel):
    """HuggingfaceModel."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
        if max_new_tokens is None:
            raise ValueError("max_new_tokens must be provided")
        self.max_new_tokens = max_new_tokens

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.device = device


        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES
        elif stop_sequences is None:
            stop_sequences = []
        self.model_name = model_name

        # -----------------------------
        # 1) Backwards-compatible Llama-2 shortcut
        # -----------------------------
        if 'llama' in model_name.lower() and not model_name.startswith(("meta-llama/", "huggyllama/")):
            # Existing logic, slightly cleaned up
            print(model_name)
            if model_name.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(load_in_8bit=True)}
                model_name_clean = model_name[:-len('-8bit')]
            else:
                kwargs = {}
                model_name_clean = model_name

            if 'Llama-2' in model_name_clean or 'Llama-3' in model_name_clean:
                base = 'meta-llama'
                # Original code appends -hf for Llama-2
                hf_name = model_name_clean + '-hf' if 'Llama-2' in model_name_clean else model_name_clean
            else:
                base = 'huggyllama'
                hf_name = model_name_clean

            model_id = f"{base}/{hf_name}"

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map="auto", token_type_ids=None
            )

            llama65b = '65b' in hf_name.lower() and base == 'huggyllama'
            llama2or3_70b = '70b' in hf_name.lower() and base == 'meta-llama'

            if ('7b' in hf_name or '13b' in hf_name) or kwargs.get('quantization_config') is not None:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    max_memory={0: '80GIB'},
                    **kwargs,
                )
            elif llama2or3_70b or llama65b:
                # keep their multi-GPU logic if you need it
                path = snapshot_download(
                    repo_id=model_id,
                    allow_patterns=['*.json', '*.model', '*.safetensors'],
                    ignore_patterns=['pytorch_model.bin.index.json']
                )
                config = AutoConfig.from_pretrained(model_id)
                with accelerate.init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(config)
                self.model.tie_weights()
                if 'chat' in hf_name:
                    max_mem = 17.5 * 4686198491
                else:
                    max_mem = 15 * 4686198491

                device_map = accelerate.infer_auto_device_map(
                    self.model.model,
                    max_memory={0: max_mem, 1: max_mem},
                    dtype='float16'
                )
                device_map = remove_split_layer(device_map)
                full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
                full_model_device_map["lm_head"] = 0

                self.model = accelerate.load_checkpoint_and_dispatch(
                    self.model, path, device_map=full_model_device_map,
                    dtype='float16', skip_keys='past_key_values')
            else:
                raise ValueError(f"Unhandled Llama variant: {model_name}")

        # -----------------------------
        # 2) Everything else: treat model_name as full HF repo id
        # -----------------------------
        else:
            # Here you can pass e.g.:
            #   meta-llama/Meta-Llama-3.1-8B-Instruct
            #   Qwen/Qwen2.5-7B-Instruct
            #   mistralai/Ministral-8B-Instruct-2410
            #   HuggingFaceTB/SmolLM3-3B-Instruct
            model_id = model_name

            # For many “modern” models you *must* set trust_remote_code=True
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                device_map="auto",
                token_type_ids=None,
                clean_up_tokenization_spaces=False,
                trust_remote_code=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                # you can add quantization here if you like:
                # quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                trust_remote_code=True,
            )

        # -----------------------------
        # 3) Stop sequences & token limit
        # -----------------------------
        # Always include EOS.
        eos = getattr(self.tokenizer, "eos_token", None)
        self.stop_sequences = list(stop_sequences)
        if eos is not None and eos not in self.stop_sequences:
            self.stop_sequences.append(eos)

        # Use model config to infer context length instead of hardcoding 4096/2048
        try:
            config = AutoConfig.from_pretrained(model_id)
            self.token_limit = getattr(config, "max_position_embeddings", 2048)
        except Exception:
            self.token_limit = 2048

    
    def predict(self, input_data, temperature, return_full=False, return_latent=False):

        if isinstance(input_data, tuple):
            logging.WARNING("INPUT IS A TUPLE.")
            input_data = input_data[0]

        inputs = self.tokenizer(input_data, return_tensors="pt").to(self.device)

        if 'llama' in self.model_name.lower() or 'falcon' in self.model_name or 'mistral' in self.model_name.lower():
            if 'token_type_ids' in inputs:  # HF models seems has changed.
                del inputs['token_type_ids']
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=len(inputs['input_ids'][0]),
                tokenizer=self.tokenizer,
                device=self.device)])
        else:
            stopping_criteria = None

        logging.debug('temperature: %f', temperature)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                do_sample=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
            )

        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True)

        if return_full:
            return full_answer

        # For some models, we need to remove the input_data from the answer.
        if full_answer.startswith(input_data):
            input_data_offset = len(input_data)
        else:
            raise ValueError('Have not tested this in a while.')

        # Remove input from answer.
        answer = full_answer[input_data_offset:]

        # Remove stop_words from answer.
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
            if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                error_msg = 'Error: Stop words not removed successfully!'
                error_msg += f'Answer: >{answer}< '
                error_msg += f'Sliced Answer: >{sliced_answer}<'
                logging.error(error_msg)

        # Remove whitespaces from answer (in particular from beginning.)
        sliced_answer = sliced_answer.strip()
        token_stop_index = self.tokenizer(full_answer[:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
        n_input_token = len(inputs['input_ids'][0])
        n_generated = token_stop_index - n_input_token

        if n_generated == 0:
            logging.warning('Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.')
            n_generated = 1

        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        if len(hidden) == 1:
            logging.warning(
                'Taking first and only generation for hidden! '
                'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s',
                n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer,
                )
            last_input = hidden[0]
        elif ((n_generated - 1) >= len(hidden)):
            # if access idx is larger/equal
            logging.error(
                'Taking last state because n_generated is too large'
                'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s, slice_answer: %s',
                n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer, sliced_answer
                )
            last_input = hidden[-1]
        else:
            last_input = hidden[n_generated - 1]

        # Then access last layer for input
        last_layer = last_input[-1]
        # Then access last token in input.
        last_token_embedding = last_layer[:, -1, :].cpu()

        if return_latent:
            # Stack second last token embeddings from all layers 
            if len(hidden) == 1:  # FIX: runtime error for mistral-7b on bioasq
                sec_last_input = hidden[0]
            elif ((n_generated - 2) >= len(hidden)):
                sec_last_input = hidden[-2]
            else:
                sec_last_input = hidden[n_generated - 2]
            sec_last_token_embedding = torch.stack([layer[:, -1, :] for layer in sec_last_input]).cpu()
    
            # Get the last input token embeddings (before generated tokens)
            last_tok_bef_gen_input = hidden[0]
            last_tok_bef_gen_embedding = torch.stack([layer[:, -1, :] for layer in last_tok_bef_gen_input]).cpu()

        # Get log_likelihoods.
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)
        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            logging.warning('Taking first and only generation for log likelihood!')
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning('Generation interrupted by max_token limit.')

        if len(log_likelihoods) == 0:
            raise ValueError

        hidden_states = (last_token_embedding,)

        if return_latent:
            hidden_states += (sec_last_token_embedding, last_tok_bef_gen_embedding)
        else:
            hidden_states += (None, None)

        return_values = (sliced_answer, log_likelihoods, hidden_states)

        return return_values

    def get_p_true(self, input_data):
        """Get the probability of the model anwering A (True) for the given input"""

        input_data += ' A'
        tokenized_prompt_true = self.tokenizer(input_data, return_tensors='pt').to(self.device)['input_ids']

        target_ids_true = tokenized_prompt_true.clone()
        # Set all target_ids except the last one to -100.
        target_ids_true[0, :-1] = -100

        with torch.no_grad():
            model_output_true = self.model(tokenized_prompt_true, labels=target_ids_true)

        loss_true = model_output_true.loss

        return -loss_true.item()

    def get_perplexity(self, input_data):
        """Get the probability of the model anwering A (True) for the given input"""

        tokenized_data = self.tokenizer(input_data, return_tensors='pt').to(self.device)['input_ids']

        with torch.no_grad():
            model_output_true = self.model(tokenized_data, labels=tokenized_data)

        perplexity = - model_output_true.loss.item()


        return perplexity
