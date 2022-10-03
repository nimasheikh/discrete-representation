import torch


def Train(
    model, 
    train_dataset, 
    test_dataset,
    learning_rate=1e-3,
    num_epochs = 10,
):
    def model_accuracy(model, test_dataset):
        model.test()
        noc = 0 
        for batch, labels in test_dataset:
            with torch.no_grad():
                logits = model(batch)
                preds = torch.argmax(logits, axis = -1)
                noc += torch.sum( preds == labels)

        return noc / len(test_dataset)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        loss_ = 0 
        for iter, (batch, labels) in enumerate(train_dataset):
            model.train()
            logits = model(batch)
            loss = loss_func( logits, labels)
            loss.backward()
            loss_ += loss.item()
            with torch.no_grad():
                optimizer.step()
                optimizer.zero_grad()
        acc = model_accuracy(model, test_dataset)
        print(f"epoch : {epoch}, acc: {acc}, loss: {loss_}")




