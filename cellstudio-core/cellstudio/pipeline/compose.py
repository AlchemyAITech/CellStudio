from typing import Any, Dict, List

from .registry import PIPELINE_REGISTRY


@PIPELINE_REGISTRY.register('Compose')
class Compose:
    """
    The main artery for running DAG-based augmentations and transformations.
    Resolves plugin dictionaries to node objects, then funnels the Universal ResultDict through sequentially.
    """
    def __init__(self, transforms: List[Dict]):
        self.transforms = []
        for t in transforms:
            if isinstance(t, dict) or hasattr(t, 'keys'):
                transform = PIPELINE_REGISTRY.build(dict(t))
                self.transforms.append(transform)
            elif callable(t):
                self.transforms.append(t)
            else:
                raise TypeError('Transform nodes must absolutely be type `callable` or `dict`.')

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stream the ResultDict continuously through the DAG.
        If any pipeline node deliberately returns None, flow halts.
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
