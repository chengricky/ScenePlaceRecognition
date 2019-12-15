from torch.utils.data import DataLoader


def loadDataSet(mode, split, whichdataset, threads, cacheBatchSize, margin):

    # Designate the dataset to train
    if whichdataset == 'pittsburgh':
        from DataSet import pittsburgh as dataset
    elif whichdataset == 'tokyo247':
        from DataSet import tokyoTM as dataset
    elif whichdataset == 'multimodal':
        from DataSet import Multimodal as dataset
    else:
        raise Exception('Unknown dataset')

    # Read image files of the designated dataset
    if mode == 'train':
        # get all of the images
        whole_train_set = dataset.get_whole_training_set()
        whole_training_data_loader = DataLoader(dataset=whole_train_set, num_workers=threads,
                                                batch_size=cacheBatchSize, shuffle=False, pin_memory=True)
        # get query-database pairs
        train_set = dataset.get_training_query_set(margin)
        print('====> Training set, query count:', len(train_set))
        print('====> Training set, database count:', train_set.dbStruct.numDb)
        # get validation dataset to choose the best trained network
        whole_test_set = dataset.get_whole_val_set()
        print('===> Evaluating on val set, query count:', whole_test_set.dbStruct.numQ)
        print('===> Evaluating on val set, database count:', whole_test_set.dbStruct.numDb)

    elif mode == 'test':
        train_set = []
        whole_train_set = []
        whole_training_data_loader = []
        if split == 'test':
            whole_test_set = dataset.get_whole_test_set()
            print('===> Evaluating on test set')
        elif split == 'test250k':
            whole_test_set = dataset.get_250k_whole_test_set()
            print('===> Evaluating on test250k set')
        elif split == 'train':
            whole_test_set = dataset.get_whole_training_set()
            print('===> Evaluating on train set')
        elif split == 'val':
            whole_test_set = dataset.get_whole_val_set()
            print('===> Evaluating on val set')
        else:
            raise ValueError('Unknown dataset split: ' + split)
        print('====> Query count:', whole_test_set.dbStruct.numQ)
        print('====> Database count:', whole_test_set.dbStruct.numDb)

    elif mode == 'cluster':
        train_set = []
        whole_test_set = []
        whole_training_data_loader = []
        whole_train_set = dataset.get_whole_training_set(onlyDB=True)

    else:
        raise Exception('Unknown mode')

    return train_set, whole_train_set, whole_training_data_loader, whole_test_set, dataset
