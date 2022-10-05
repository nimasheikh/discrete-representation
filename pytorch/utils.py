import torch

def find_image_for_representation(
    model,
    init,
    representation,
    loss_func=torch.nn.BCELoss(),
    num_iter=10, 
    lr = 1e-2
):
    
    model.train()
    img_ = torch.nn.Parameter(init ,requires_grad = True)
    optimizer = torch.optim.Adam([img_], lr = lr)
    for i in range(num_iter):
        rep_ = model.representation(img_)
        loss = loss_func(rep_, representation)
        loss.backward()
        with torch.no_grad():
            optimizer.step()
            optimizer.zero_grad()
    
        print(f"iter: {i}, loss: {round(loss.item(), 4)}, rep_: {rep_}")
    
    return img_

def find_advs_img(
    model,
    init,
    target_class,
    loss_func=torch.nn.CrossEntropyLoss(),
    num_iter=10,
    lr = 1e-2
):
    
    model.train()
    img_ = torch.nn.Parameter(init, requires_grad=True)
    optimizer = torch.optim.Adam([img_], lr=lr)
    for i in range(num_iter):
        rep_ = model(img_)
        loss = loss_func(rep_, target_class)
        loss.backward()
        with torch.no_grad():
            optimizer.step()
            optimizer.zero_grad()
    
        print(f"iter: {i}, loss: { round(loss.item(), 4)}, target_class: {rep_.argmax()}" )
    
    return init

