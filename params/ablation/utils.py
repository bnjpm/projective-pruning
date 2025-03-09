import torch


def top1_correct(output, target) -> int:
    pred = output.argmax(dim=1)
    correct = pred.eq(target).sum().item()
    return correct


def top5_correct(output, target) -> int:
    _, pred = output.topk(5, dim=1)
    correct = pred.eq(target.view(-1, 1).expand_as(pred)).sum().item()
    return correct


def get_eval(criterion, dataloader, device):
    @torch.no_grad()
    def evaluate_fn(model):
        model.eval()
        num_correct1 = 0
        num_correct5 = 0
        num_processed = 0
        total_loss = 0

        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            output = model(X)

            total_loss += criterion(output, y).item()
            num_correct1 += top1_correct(output, y)
            num_correct5 += top5_correct(output, y)
            num_processed += X.shape[0]

        acc1 = num_correct1 / num_processed
        acc5 = num_correct5 / num_processed
        loss = total_loss

        return acc1, acc5, loss

    return evaluate_fn
