import sys

# __import__('pkg_resources').declare_namespace(__name__)
# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'asr_recipe'))
# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nlu_recipe'))


import os
__path__.extend([os.path.join(__path__[0], '..', '..', 'asr_recipe', 'recipe')])
__path__.extend([os.path.join(__path__[0], '..', '..', 'nlu_recipe')])
__path__.extend([os.path.join(__path__[0], '..', '..', 'dialdoc_recipe')])
