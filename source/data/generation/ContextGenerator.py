from typing import List
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser, Namespace
from source.data.generation.PromptGenerator import PromptGenerator


class ContextGenerator :

    def __init__(
            self,
            LLM_key : str,
            prompt_generator: PromptGenerator,
            use_fast_tokenizer : bool = True,
            batch_size : int = 32,
            generation_max_length : int = 512,
            use_cuda : bool = False,
            is_split_into_words : bool = False,
            is_pretokenized : bool = False,

    ) :

        # batch size
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.is_split_into_words = is_split_into_words

        assert 1 <= generation_max_length <= 512, f"Generation max length should be nbetween 1 and 512."
        self.generation_max_length = generation_max_length

        # init model for generation
        print("## LOADING MODEL...", end="")
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_key,
            use_auth_token="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",
            device_map='auto'
        )
        print("DONE")

        print("## FITTING MODEL TO GPU...", end="")
        if use_cuda:
            self.model.cuda()
        print("DONE")

        # init tokenizer
        print("## LOADING TOKENIZER...", end="")
        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_key,
            use_auth_token="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",
            is_split_into_words=self.is_split_into_words
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("DONE")

        self.prompt_generator = prompt_generator

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


    def generate(self, input_texts : List[str])  :
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

            #i = 0

            for texts, original_texts in tqdm(generator, total=len(input_texts) // self.batch_size) :


                # tokenizzing text
                encoded_input = self.tokenizer(
                    text = texts,
                    return_tensors='pt',
                    padding=True,
                    is_split_into_words = False
                )

                if self.use_cuda :
                    encoded_input = self.move_data_GPU(encoded_input)

                # generation
                generate_ids  = self.model.generate(
                    encoded_input['input_ids'],
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
                current_output = [  # removing prompt
                    generated_answer.replace(text, '').strip() for generated_answer, text in zip(current_output, texts)
                ]
                current_output = [  # removing \n
                    generated_answer.replace("\n", '.').strip() for generated_answer, text in zip(current_output, texts)
                ]

                # post process for easier statistics
                for  text, generated_answer in zip(original_texts, current_output) :
                    outputs_post_process.append(
                        { "text" : text, "generated_context" : generated_answer }
                    )

                outputs += current_output

                # if i >= 3 :
                #     break
                #
                # i += 1


            return outputs, outputs_post_process

    @staticmethod
    def add_specific_args(parent_parser : ArgumentParser) -> ArgumentParser :

        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--LLM_key", type = str, default = 'meta-llama/Llama-2-7b-chat-hf')
        parser.add_argument("--use_fast_tokenizer", action='store_true', default = False)
        parser.add_argument("--use_cuda", action='store_true', default = False)
        parser.add_argument("--is_split_into_words", action='store_true', default = False)
        parser.add_argument("--batch_size", type = int, default=32)
        parser.add_argument("--generation_max_length", type = int, default=256)

        return parser
