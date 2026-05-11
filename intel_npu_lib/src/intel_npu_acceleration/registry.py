from typing import Dict, Any, Callable, Optional


class OpRegistry:
    _function_converters: Dict[Any, Callable] = {}
    _method_converters: Dict[str, Callable] = {}
    _module_converters: Dict[Any, Callable] = {}

    @classmethod
    def register_function(cls, *targets):
        def decorator(func):
            for t in targets:
                cls._function_converters[t] = func
            return func

        return decorator

    @classmethod
    def register_method(cls, *names):
        def decorator(func):
            for n in names:
                cls._method_converters[n] = func
            return func

        return decorator

    @classmethod
    def register_module(cls, *types):
        def decorator(func):
            for t in types:
                cls._module_converters[t] = func
            return func

        return decorator

    @classmethod
    def get_function(cls, target) -> Optional[Callable]:
        return cls._function_converters.get(target)

    @classmethod
    def get_method(cls, name: str) -> Optional[Callable]:
        return cls._method_converters.get(name)

    @classmethod
    def get_module(cls, module_type) -> Optional[Callable]:
        return cls._module_converters.get(module_type)
