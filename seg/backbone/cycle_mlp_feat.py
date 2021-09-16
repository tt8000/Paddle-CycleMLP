from cycle_mlp import CycleNet, CycleMLP


# For dense prediction tasks only
class CycleMLP_B1_feat(CycleNet):
    def __init__(self, **kwargs):
        transitions = [True, True, True, True]
        layers = [2, 2, 4, 2]
        mlp_ratios = [4, 4, 4, 4]
        embed_dims = [64, 128, 320, 512]
        super(CycleMLP_B1_feat, self).__init__(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                                                mlp_ratios=mlp_ratios, mlp_fn=CycleMLP, fork_feat=True)

class CycleMLP_B2_feat(CycleNet):
    def __init__(self, **kwargs):
        transitions = [True, True, True, True]
        layers = [2, 3, 10, 3]
        mlp_ratios = [4, 4, 4, 4]
        embed_dims = [64, 128, 320, 512]
        super(CycleMLP_B2_feat, self).__init__(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                                                mlp_ratios=mlp_ratios, mlp_fn=CycleMLP, fork_feat=True)


class CycleMLP_B3_feat(CycleNet):
    def __init__(self, **kwargs):
        transitions = [True, True, True, True]
        layers = [3, 4, 18, 3]
        mlp_ratios = [8, 8, 4, 4]
        embed_dims = [64, 128, 320, 512]
        super(CycleMLP_B3_feat, self).__init__(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                                                mlp_ratios=mlp_ratios, mlp_fn=CycleMLP, fork_feat=True)


class CycleMLP_B4_feat(CycleNet):
    def __init__(self, **kwargs):
        transitions = [True, True, True, True]
        layers = [3, 8, 27, 3]
        mlp_ratios = [8, 8, 4, 4]
        embed_dims = [64, 128, 320, 512]
        super(CycleMLP_B4_feat, self).__init__(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                                                mlp_ratios=mlp_ratios, mlp_fn=CycleMLP, fork_feat=True)


class CycleMLP_B5_feat(CycleNet):
    def __init__(self, **kwargs):
        transitions = [True, True, True, True]
        layers = [3, 4, 24, 3]
        mlp_ratios = [4, 4, 4, 4]
        embed_dims = [96, 192, 384, 768]
        super(CycleMLP_B5_feat, self).__init__(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                                                mlp_ratios=mlp_ratios, mlp_fn=CycleMLP, fork_feat=True)