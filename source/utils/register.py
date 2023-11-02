from functools import wraps


"""
   Defining register to store building blocks. This is later used by the argument parser.
"""
Registers = {
    "LOSSES"          : dict(),
    "MODULES"         : dict(),
    "MODELS"          : dict(),
    "DATASETS"        : dict(),
    "TASKS"           : dict(),
    "CALLBACKS"       : dict(),
    "METRICS"         : dict()
}


def register(type : str) :
    """
    Decorator used to store a given elements in its register
    :param type: register to store the element in
    """
    if type not in Registers.keys() :
        raise ValueError(f"Unsupported type : {type}. Available type : {list(Registers)}")

    def _register(cls) :
        @wraps(cls)
        def _register_elt(cls) :
            for name in getattr(cls, "_names", []) + [cls.__name__] :
                Registers[type][name] = cls
            return cls
        return _register_elt(cls)
    return _register


class dataset:
    def __init__(self, fn):
        self.fn = fn

    def __set_name__(self, owner, name):
        # do something with owner, i.e.
        print(f"decorating {self.fn} and using {owner}")
        self.fn.class_name = owner.__name__

        # then replace ourself with the original method
        setattr(owner, name, self.fn)


def dataset(dataset : str) :
    """
    Decorator used to assign a sample creation method to a dataset
    """

    if dataset not in Registers['DATASETS'].keys() :
        raise ValueError(f"Unsupported dataset : {dataset}. Available datasets : {Registers['DATASETS'].keys()}")

    def wrapper(method) :
        @wraps(method)
        def _wrapper(method) :
            method_class = method.im_class
            method_class.sample_creation

