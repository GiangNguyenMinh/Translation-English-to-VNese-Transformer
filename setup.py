import os
import torch
import time
import copy

def train_model(model, dataLoader, criterion, optimizer, n_epochs, dataset_size, device, create_mask):

    since = time.time()
    iter_start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        print('Epoch: {}/{}'.format(epoch, n_epochs))
        print('--'*20)
        iteration = 1

        model.to(device)

        for phase in ['train', 'val']:
            if phase == 'train':
                print('Training')
                model.train()
            else:
                print('--'*20)
                print('Evaluating')
                model.eval()

            running_loss = 0.0
            running_correct = 0.0

            for src, tgt in dataLoader[phase]:
                src = src.to(device)
                tgt = tgt.to(device)

                tgt_input = tgt[:-1, :]
                tgt_out = tgt[1:, :]
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask,
                                   tgt_padding_mask, src_padding_mask)
                    _, preds = torch.max(logits.reshape(-1, logits.shape[-1]), dim=1)
                    loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))  # (T.B)x vocab_tgt and tgt size

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    if iteration % 100 == 0:
                        iter_end = time.time()
                        duration = iter_end - iter_start
                        print('Epoch: {}/{} || Iteration {} || loss: {:.4f} || Duration 100 iter: {:.0f}m {:.0f}s)'
                              .format(epoch, n_epochs, iteration, loss, duration // 60, duration % 60))
                        iter_start = time.time()

                running_loss += loss.item()
                running_correct += torch.sum(preds == tgt_out.reshape(-1))
                iteration += 1

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_correct / dataset_size[phase]

            print('Loss in Epoch {}/{} is {:.4f}'.format(epoch+1, n_epochs, epoch_loss))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Traning complete in: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best accurency: ', best_acc)

    if not os.path.exists('./weight'):
        os.mkdir('./weight')
    torch.save(best_model_wts, os.path.join('./weight', 'translation.pth'))
    model.load_state_dict(best_model_wts)
    return model

