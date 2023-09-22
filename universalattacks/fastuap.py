from .universalattack import UniversalAttack

class FastUAP(UniversalAttack):
    r'''
    FastUAP method in paper "Fast-UAP: An Algorithm for Speeding up Universal Adversarial Perturbation Generation with Orientation of Perturbation Vectors"
    [https://arxiv.org/ftp/arxiv/papers/1911/1911.01172.pdf]
    '''
    def __init__(self, model):
        super().__init__("FastUAP", model)