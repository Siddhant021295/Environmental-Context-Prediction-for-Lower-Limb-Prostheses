def training(train_losslist,model,train_loader,flag_cuda,criterion,optimizer,len_train):
    train_loss = 0.0
    for data, target in train_loader:
        # Moving tensors to GPU if CUDA is available
        if flag_cuda:
            data, target = data.cuda(), target.cuda()
        # Clearing the gradients of all optimized variables
        optimizer.zero_grad()

        output = model(data)
        # Calculating the batch loss
        # print(type(output),output, type(target),target)
        loss = criterion(output, target)
        # Backward pass: compute gradient of loss with respect to parameters
        loss.backward()
        # Perform a single optimization step (parameter update)
        optimizer.step()
        # Update training loss
        train_loss += loss.item()*data.size(0)

    # Calculating average training losses
    train_loss = train_loss/len_train
    train_losslist.append(train_loss)

    return train_losslist,train_loss,model