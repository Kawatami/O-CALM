from typing import Tuple, List

class PromptGenerator :
    """
    Object used to generate prompt with context
    """

    _default_prompt : str = \
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, " \
        "while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, " \
        "dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature." \
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something " \
        "not correct. If you don't know the answer to a question, please don't share false information."

    _default_instruction_tag : Tuple[str, str] = ("[INST]", "[/INST]")
    _default_system_tags : Tuple[str, str] = ("<<SYS>>", "<</SYS>>")

    def __init__(
            self,
            instruction_tag : Tuple[str, str] = None,
            system_tags : Tuple[str, str] = None,
            prompt : str = None

    ) :
        """
        Instantiate the prompt generator.
        :param instruction_tag:
        :param system_tags:
        :param prompt:
        """

        self.instruction_tag = PromptGenerator._default_instruction_tag if instruction_tag is None else instruction_tag
        self.system_tags = PromptGenerator._default_system_tags if system_tags is None else system_tags
        self.prompt = PromptGenerator._default_prompt if prompt is None else prompt

    def __call__(self, input_text : List[str]) :

        return [
            f"{self.instruction_tag[0]} {self.system_tags[0]} {self.prompt} {self.system_tags[1]} {text} {self.instruction_tag[1]}"
            for text in input_text
        ]

