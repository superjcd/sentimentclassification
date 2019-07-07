import time
import torch



def train(net, optimizer, loss_func, train_iter, val_iter, compute_val,
          device, epoches, load_model_dir, save_model_dir):
    '''

    :param net:
    :param optimizer:
    :param loss_func:
    :param train_iter:
    :param val_iter:
    :param compute_val:
    :param device:
    :param epoches:
    :param load_model_dir:
    :param save_model_dir:
    :return:
    '''
    print(f'>>>We are gonna tranning {net.__class__.__name__} with epoches of {epoches}<<<')
    net = net.to(device)
    if load_model_dir:
        net.load_state_dict(torch.load(load_model_dir))
    batch_count = 0
    for epoch in range(epoches):
        print(f'=>we are training epoch[{epoch+1}]...<=')
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for iter_num, batch in enumerate(train_iter):
            X = batch.data.to(device)
            y = batch.label.squeeze(0).to(device)
            score = net(X)
            try:
                l = loss_func(score, y)  # 一定是score在前， y在后！
            except:
                print('error occured!!')
                print(f'score shape:{score.shape}; y shape:{y.shape}; y:{y}')
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (score.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            train_acc = train_acc_sum / n
            if (iter_num+1) % 10 == 0:
                print("Train accuracy now is %.1f" % (round(train_acc, 3)*100)+'%')
                ## 计算validation score
                if compute_val:
                    net.eval()
                    val_data = next(iter(val_iter))
                    val_X = val_data.data.to(device)
                    val_y = val_data.label.squeeze(0).to(device)
                    val_score = net(val_X)
                    val_acc = (val_score.argmax(dim=1) == val_y).sum().cpu().item()/len(val_y)
                    print("Val accuracy of one batch is %.1f " % (round(val_acc, 3)*100)+'%')
                    net.train()
                print('*'*25)
        if (epoch+1) % 5 ==0 and save_model_dir:
            print(f'saving model into => {save_model_dir}')
            torch.save(net.state_dict(), save_model_dir)
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc, time.time() - start))



