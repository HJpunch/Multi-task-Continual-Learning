from .pass_trainer_joint import GeneralTransformerTrainer_joint

class TrainerFactory(object):
    def __init__(self):
        super(TrainerFactory, self).__init__()
        self.snr_net = ['resnet_ibn50a_snr', 'resnet_ibn101a_snr',
                        'resnet_ibn50a_snr_spatial', 'resnet_ibn101a_snr_spatial']
        self.mgn_net = ['mgn']
        self.clothes_net = ['resnet_ibn101a_two_branch', 'resnet_ibn50a_two_branch', 'transformer', 'transformer_kmeans']
        self.transreid_two_branch = ['deit_small_patch16_224_TransReID','deit_small_patch16_224_TransReID_mask']
        self.transreid_two_branch_aug = ['deit_small_patch16_224_TransReID_aug','deit_small_patch16_224_TransReID_mask_aug']
        self.transformer = ['transformer_dualattn']
        self.transformer_t2i = ['transformer_dualattn_t2i']
        self.transformer_joint = ['transformer_dualattn_joint']

    def create(self, name, *args, **kwargs):
        if name in self.transformer_joint:
            return GeneralTransformerTrainer_joint(*args, **kwargs)
