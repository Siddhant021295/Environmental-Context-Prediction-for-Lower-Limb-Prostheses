import torch
def evaluate_confusion_matrix(model,test_loader,confusion,flag_cuda):
    with torch.no_grad():
        for data, target in test_loader:
            # Moving tensors to GPU if CUDA is available
            if flag_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            _, preds = torch.max(output, 1)
            for t, p in zip(target.view(-1), preds.view(-1)):
                confusion[t.long(), p.long()] += 1
            #print(confusion)
    
    accuracy = 0
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
        accuracy += confusion[i][i]
    
    accuracy /= n_categories

    # Displaying the average accuracy
    print('Average Macro Accuracy = {:.2f}\n'.format(accuracy))
    return confusion