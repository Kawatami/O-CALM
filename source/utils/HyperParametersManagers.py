class SingletonMeta(type):
    """
    Defining the metaclass for singleton design pattern.
    More info here : https://refactoring.guru/design-patterns/singleton/python/example
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class HyperParametersManager(dict ,metaclass=SingletonMeta) :
    """
    This class intend to store and make accessible from anywhere in the code any potential information processed at
    runtime such as the number of training steps for example as it may depend on the number of GPU available,
    gradient accumulation etc...

    The class is a simple dictionary combined with the singleton design pattern.

     ```python
        >>> from source.utils.HyperParametersManagers import HyperParametersManager
        >>> # call the dictionary from anywhere
        >>> HyperParametersManager()['KEY_EXEMPLE'] = 'VALUE_EXEMPLE'
        >>> # read from anywhere
        >>> HyperParametersManager()['KEY_EXEMPLE']
        >>> 'VALUE_EXEMPLE'
    ```



    """
    pass

