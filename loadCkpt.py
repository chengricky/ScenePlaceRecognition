from os.path import join, isfile
import torch


def loadckpt(ckpt, resume, start_epoch, mode, optLoaded, nGPU, device, model, withAttention):

    if ckpt == 'latest':
        resume_ckpt = join(resume, 'checkpoint.pth.tar')
    elif ckpt == 'best':
        resume_ckpt = join(resume, 'model_best.pth.tar')
    else:
        raise Exception("Undefined ckpt type")

    if isfile(resume_ckpt):
        print("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        start_epoch = min(checkpoint['epoch'], start_epoch)
        best_metric = checkpoint['best_score']

        if mode == 'cluster':
            state_dict = {k: v for k, v in
                          checkpoint['state_dict'].items() if 'pool' not in k}
            if optLoaded.nGPU == 1:
                state_dict = {str.replace(k, 'encoder.', 'encoder.module.'): v for k, v in
                              state_dict.items()}  # add 'module.'
            model.load_state_dict(state_dict, strict=True)
        else:
            if optLoaded.nGPU == 1 and nGPU > 1:
                state_dict = {str.replace(k, 'encoder.', 'encoder.module.'): v for k, v in
                              checkpoint['state_dict'].items()}
                state_dict = {str.replace(k, 'pool.', 'pool.module.'): v for k, v in
                              state_dict.items()}  # add 'module.'
            elif optLoaded.nGPU > 1 and nGPU == 1:
                state_dict = {str.replace(k, 'encoder.module.', 'encoder.'): v for k, v in
                              checkpoint['state_dict'].items()}
                state_dict = {str.replace(k, 'pool.module.', 'pool.'): v for k, v in
                              state_dict.items()}  # remove 'module.'
            else:
                state_dict = checkpoint['state_dict']
            state_dict_encoder = {k: v for k, v in state_dict.items() if 'encoder' in k}
            state_dict_encoder = {str.replace(k, 'encoder.', ''): v for k, v in state_dict_encoder.items()}
            model.encoder.load_state_dict(state_dict_encoder, strict=True)

            state_dict_pool = {k: v for k, v in state_dict.items() if 'pool' in k}
            state_dict_pool = {str.replace(k, 'pool.', ''): v for k, v in state_dict_pool.items()}
            model.pool.load_state_dict(state_dict_pool, strict=True)

            if withAttention:
                state_dict_pool = {k: v for k, v in state_dict.items() if 'attention' in k}
                state_dict_pool = {str.replace(k, 'attention.', ''): v for k, v in state_dict_pool.items()}
                model.attention.load_state_dict(state_dict_pool, strict=True)

        model = model.to(device)
        # if opt.mode == 'train':
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_ckpt, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_ckpt))

    return model, start_epoch, best_metric
