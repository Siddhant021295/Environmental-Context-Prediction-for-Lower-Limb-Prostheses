def validation(valid_losslist,model,test_loader):
    valid_loss = 0.0
    for data, target in test_loader:
        # Moving tensors to GPU if CUDA is available
        if flag_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)

  # Calculating average validation losses
    valid_loss = valid_loss/len_test
    valid_losslist.append(valid_loss)
    return valid_losslist,valid_loss,model
