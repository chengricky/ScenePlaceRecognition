import numpy as np
import torch
from torch.utils.data import DataLoader
from os import path
from UnifiedModel import Backbone as mdl


def generate(rv, opt, path_dir):
    numDb = rv.whole_test_set.dbStruct.numDb
    subdir = ['reference', 'query']
    test_data_loader = DataLoader(dataset=rv.whole_test_set, num_workers=opt.threads,
                                  batch_size=opt.cacheBatchSize, shuffle=False, pin_memory=True)
    size = None
    rv.model.eval()
    with torch.no_grad():
        print('====> Generating Descriptors')
        for iteration, (rgb, ir, indices) in enumerate(test_data_loader, 1):
            # GLOBAL Decs
            rgb = rgb.to(rv.device)
            image_encoding = rv.model.encoder(rgb)
            if opt.withAttention:
                image_encoding = rv.model.attention(image_encoding)
                vlad_encoding = rv.model.pool(image_encoding)
            else:
                vlad_encoding = rv.model.pool(image_encoding)

            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch-RGB ({}/{})".format(iteration, len(test_data_loader)), flush=True)

            for i in range(vlad_encoding.size()[0]):
                idx = int(indices[i]) % numDb
                savepth = path.join(path_dir+subdir[int(indices[i]) // numDb], str(idx).zfill(6)+'.rgb.npy')
                np.save(savepth, vlad_encoding[i, :].detach().cpu().numpy())

            mdl.hook_features.clear()
            del rgb, image_encoding, vlad_encoding

            # LOCAL Decs
            ir = ir.to(rv.device)
            _t = rv.model.encoder(ir)
            local_feat = mdl.hook_features[-1]
            size = local_feat.shape[1:]

            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch-IR ({}/{})".format(iteration, len(test_data_loader)), flush=True)

            for j in range(local_feat.shape[0]):
                idx = int(indices[j]) % numDb
                savepth = path.join(path_dir+subdir[int(indices[j]) // numDb], str(idx).zfill(6)+'.ir.npy')
                np.save(savepth, local_feat[j, :, :, :])
            mdl.hook_features.clear()
            del ir, local_feat, _t

    with open(path.join(path_dir,"paras.txt"), 'w') as fw:
        for ele in size:
            fw.write(str(ele)+'\n')

    del test_data_loader




