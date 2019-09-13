

def test(eval_set, epoch=0, write_tboard=False):
    # TODO what if features dont fit in memory?
    test_data_loader = DataLoader(dataset=eval_set, num_workers=opt.threads,
                                  batch_size=opt.cacheBatchSize, shuffle=False,
                                  pin_memory=True)

    model.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = encoder_dim
        if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
        dbFeat = np.empty((len(eval_set), pool_size))

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            input = input.to(device)
            image_encoding = model.encoder(input)
            if opt.withAttention:
                image_encoding_a = model.attention(image_encoding)
                vlad_encoding = model.pool(image_encoding_a)
            else:
                vlad_encoding = model.pool(image_encoding)

            dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, len(test_data_loader)), flush=True)

            del input, image_encoding, vlad_encoding
    del test_data_loader

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[eval_set.dbStruct.numDb:].astype('float32')
    dbFeat = dbFeat[:eval_set.dbStruct.numDb].astype('float32')

    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    print('====> Calculating recall @ N')
    n_values = [1, 5, 10, 20]
    if opt.dataset.lower() == 'tokyo247':
        n_values = [10, 50, 100, 200]

    _, predictions = faiss_index.search(qFeat, max(n_values))
    predictionNew = []
    if opt.dataset.lower() == 'tokyo247':
        for idx, pred in enumerate(predictions):
            keep = [True for pidx in pred if eval_set.dbStruct.dbTimeStamp[pidx] != eval_set.dbStruct.qTimeStamp[idx]]
            # or (not (eval_set.dbStruct.utmDb[pidx] == eval_set.dbStruct.utmQ[idx]).all())]
            pred_keep = [pred[idxiii] for idxiii, iii in enumerate(keep) if iii is True]
            predictionNew.append(pred_keep[:max(n_values) // 10])
        predictions = predictionNew
        n_values = [1, 5, 10, 20]
    # elif opt.dataset.lower() == 'pittsburgh':
    #     for idx, pred in enumerate(predictions):
    #         keep = [True for pidx in pred if not (eval_set.dbStruct.utmDb[pidx] == eval_set.dbStruct.utmQ[idx]).all()]
    #         pred_keep = [pred[idxiii] for idxiii, iii in enumerate(keep) if iii is True]
    #         predictionNew.append(pred_keep[:max(n_values)//10])
    #     predictions = predictionNew
    #     n_values = [1, 5, 10, 20]

    # for each query get those within threshold distance
    gt = eval_set.getPositives()

    correct_at_n = np.zeros(len(n_values))
    gtValid = 0
    # TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        # print(pred)
        # print(gt[qIx])
        if gt[qIx].size:
            gtValid += 1
            for i, n in enumerate(n_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], gt[qIx])):
                    correct_at_n[i:] += 1
                    break
    recall_at_n = correct_at_n / gtValid  # eval_set.dbStruct.numQ

    recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard: writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch)

    return recalls