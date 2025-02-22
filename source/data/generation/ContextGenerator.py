import pathlib
from typing import List, Optional
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser, Namespace
from source.data.generation.PromptGenerator import PromptGenerator
import os

class ContextGenerator :
    """

    """


    def __init__(
            self,
            LLM_key : str,
            prompt_generator: PromptGenerator,
            batch_size : int = 32,
            generation_max_length : int = 512,
            use_cuda : bool = False,
            is_split_into_words : bool = False,
            LLM_load_locally : Optional[str] = None,
            tokenizer_load_locally : Optional[str] = None,
            display_generated_output : bool = False
    ) :
        """

        """

        # batch size
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.is_split_into_words = is_split_into_words
        self.display_generated_output = display_generated_output

        assert 1 <= generation_max_length <= 512, f"Generation max length should be nbetween 1 and 512."
        self.generation_max_length = generation_max_length

        # init model for generation
        print("## LOADING MODEL...", end="")
        if LLM_load_locally is None :
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_key,
                use_auth_token="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",
                device_map='auto'
            )
        else :
            self.model = self.load_model_locally(LLM_load_locally)
        print("DONE")

        print("## FITTING MODEL TO GPU...", end="")
        if use_cuda:
            self.model.cuda()
        print("DONE")

        # init tokenizer
        print("## LOADING TOKENIZER...", end="")
        if tokenizer_load_locally is None :

            self.tokenizer = AutoTokenizer.from_pretrained(
                LLM_key,
                token="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",
                is_split_into_words=self.is_split_into_words
            )
        else :
            self.tokenizer = self.load_tokenizer_locally(tokenizer_load_locally)

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("DONE")

        self.prompt_generator = prompt_generator

    def load_model_locally(self, LLM_load_locally) :
        """
        Load the transformers model from local files given the env variable MODEL_FILES is defined.
        :param transformer_local_load: subdirectory holding the model files
        :return : the loaded model
        """

        # sanity check
        if "MODEL_FILES" not in os.environ:
            raise EnvironmentError(f"No $MODEL_FILES env variable have been defined. Make sure to set to "
                                   f"model file location in order to load them locally.")

        # creating path
        path_model = pathlib.Path(os.environ['MODEL_FILES']) / LLM_load_locally

        # checking model directory
        if not path_model.exists() :
            raise FileNotFoundError(f"Model file not found. Directory {path_model} not found.")

        # loading model
        model = AutoModelForCausalLM.from_pretrained(
            path_model,
            device_map='auto'
        )

        return model

    def load_tokenizer_locally(self, tokenizer_load_locally):
        """
        Load the transformers tokenizer from local files given the env variable MODEL_FILES is defined.
        :param transformer_local_load: subdirectory holding the model files
        :return : the loaded model
        """

        # sanity check
        if "MODEL_FILES" not in os.environ:
            raise EnvironmentError(f"No $MODEL_FILES env variable have been defined. Make sure to set to "
                                   f"model file location in order to load them locally.")

        # creating path
        path_model = pathlib.Path(os.environ['MODEL_FILES']) / tokenizer_load_locally

        # loading model
        model = AutoTokenizer.from_pretrained(path_model)

        return model

    def move_data_GPU(self, encoded_input) :

        for key, value in encoded_input.items() :

            if isinstance(encoded_input[key], torch.Tensor) :
                encoded_input[key] = encoded_input[key].cuda()

        return encoded_input

    def create_generator(self, input_texts : List[str]) :
        """
        Create a generator from a given list of raw text
        :param input_texts:
        :param prompt:
        :return:
        """

        for i in range(0, len(input_texts), self.batch_size) :
            original_texts = input_texts[i:(i+self.batch_size)]

            # merging tokens
            if self.is_split_into_words :
                original_texts = [" ".join(sequence) for sequence in original_texts]

            prompted_text = self.prompt_generator(original_texts)

            yield prompted_text, original_texts

    def generate(self, input_texts : List[str], limit = None)  :
        """
        Generate a context given a text and an input prompt
        :param input_text:
        :param prompt:
        :return:
        """

        outputs = []
        outputs_post_process = []

        # create generator
        generator = self.create_generator(input_texts)

        with torch.no_grad() :

            i = 0

            for texts, original_texts in tqdm(generator, total=len(input_texts) // self.batch_size) :


                # tokenizzing text
                encoded_input = self.tokenizer(
                    text = texts,
                    return_tensors='pt',
                    padding=True,
                    is_split_into_words = False
                )

                #if self.use_cuda :
                encoded_input = self.move_data_GPU(encoded_input)



                # generation
                generate_ids  = self.model.generate(
                    encoded_input['input_ids'],
                    attention_mask = encoded_input['attention_mask'],
                    # num_beams=5,
                    max_new_tokens=self.generation_max_length,
                    # early_stopping=True,
                    # temperature=0.6,
                    # repetition_penalty=1.1
                )

                # decoding
                current_output = self.tokenizer.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

                # cleaning output
                # #current_output = [  # removing prompt
                #     generated_answer.replace(text, '').strip() for generated_answer, text in zip(current_output, texts)
                # ]
                # current_output = [  # removing \n
                #     generated_answer.replace("\n", '.').strip() for generated_answer, text in zip(current_output, texts)
                # ]

                # post process for easier statistics
                for  text, generated_answer in zip(original_texts, current_output) :
                    outputs_post_process.append(
                        { "text" : text, "generated_context" : generated_answer }
                    )

                    if self.display_generated_output :
                        print("====")
                        print(text)
                        print("----")
                        print(generated_answer)

                outputs += current_output

                if limit is not None and i >= limit :
                    break

                i += 1


            return outputs, outputs_post_process

    @staticmethod
    def add_specific_args(parent_parser : ArgumentParser) -> ArgumentParser :

        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--LLM_key", type = str, default = 'meta-llama/Llama-2-7b-chat-hf')
        parser.add_argument("--LLM_load_locally", type = str, default=None)
        parser.add_argument("--tokenizer_load_locally", type = str, default=None)
        parser.add_argument("--use_cuda", action='store_true', default = False)
        parser.add_argument("--is_split_into_words", action='store_true', default = False)
        parser.add_argument("--display_generated_output", action='store_true', default = False)
        parser.add_argument("--batch_size", type = int, default=32)
        parser.add_argument("--generation_max_length", type = int, default=256)

        return parser
